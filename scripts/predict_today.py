"""
快速预测今日区间和信号 + 期权策略建议

只训练最后一折 DL Range 模型, 预测最新日期.
结合 Regime + RV + Hybrid Band 输出交易信号.
触发信号时, 基于 EOD 快照给出不同风险等级的期权策略建议.

用法:
    conda activate gold
    python scripts/predict_today.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from glob import glob
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang HK",
                                "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = "/Users/yhdong/Gold"
sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.dl_fair_value import select_features
from src.models.dl_range_predictor import DLRangePredictor
from src.models.regime_classifier import RegimeClassifier
from src.models.analysis_method_compare import (
    build_band, generate_v2_signals, compute_rv_pctile,
    compute_tp_indicators,
)


def load_latest_eod_snapshot():
    """加载最新的 EOD 期权快照."""
    snap_dir = os.path.join(PROJECT_ROOT, "data", "raw", "options_history")
    snap_dates = sorted(glob(os.path.join(snap_dir, "202*", "eod_full.parquet")))
    if not snap_dates:
        return None, None
    latest_path = snap_dates[-1]
    snap_date = os.path.basename(os.path.dirname(latest_path))
    return pd.read_parquet(latest_path), snap_date


def print_strategy_recommendations(signal_type, gld_price, pred_u_pct, pred_l_pct,
                                   rv_pctile, eod_df, snap_date):
    """
    根据信号类型, 输出不同风险等级的期权策略建议.

    策略分三档:
      稳健: 低杠杆, 高胜率, 适合大仓位
      中性: 平衡风险收益
      激进: 高杠杆, 高收益但高风险, 小仓位
    """
    print(f"\n  {'═' * 56}")
    print(f"  期权策略建议 (基于 {snap_date} EOD快照)")
    print(f"  {'═' * 56}")

    # 模型预测的5日目标价
    target_up = gld_price * (1 + pred_u_pct / 100)
    target_lo = gld_price * (1 + pred_l_pct / 100)

    if signal_type == "BUY_CALL":
        _print_buy_call_strategies(gld_price, target_up, target_lo,
                                   pred_u_pct, rv_pctile, eod_df)
    elif signal_type == "SELL_PUT":
        _print_sell_put_strategies(gld_price, target_up, target_lo,
                                   pred_u_pct, rv_pctile, eod_df)
    elif signal_type == "EXIT":
        _print_exit_strategies(gld_price, eod_df)

    # 风控提示
    print(f"\n  ── 风控要点 ──")
    print(f"  · 单笔仓位: 总资金的 2-5% (稳健) / 5-10% (中性) / ≤5% (激进)")
    print(f"  · 平仓条件: bp>0.90 或 Regime退出Bull 或 持仓10天")
    print(f"  · 止盈拐点: 涨幅>2%后回撤≥1.5% → 平仓")
    print(f"  · 模型覆盖率: 74% (有26%概率突破预测区间)")


def _find_options(eod_df, option_type, strike_range, dte_range):
    """从EOD快照中筛选期权."""
    if eod_df is None:
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


def _fmt_contract(row):
    """格式化合约显示."""
    exp = pd.Timestamp(row["strike_time"]).strftime("%m/%d")
    strike = row["option_strike_price"]
    otype = "C" if row["option_type"] == "CALL" else "P"
    return f"GLD {exp} ${strike:.0f}{otype}"


def _print_buy_call_strategies(gld_price, target_up, target_lo,
                                pred_u_pct, rv_pctile, eod_df):
    """BUY CALL 信号的三档策略."""
    print(f"\n  信号: BUY CALL (Bull + 低位 + 正常波动率)")
    print(f"  预期: 5日上行 +{pred_u_pct:.1f}% → ${target_up:.0f}")

    # ── 策略A: 稳健 ──
    print(f"\n  ┌─ A. 稳健策略: 买入 ITM Call ─────────────────────")
    print(f"  │  目标: Delta≈0.70, DTE 30-45天, 高内在价值")
    print(f"  │  优点: 时间损耗小, 走势跟踪紧密, 下跌有内在价值缓冲")
    print(f"  │  风险: 权利金较高, 杠杆较低")
    itm_calls = _find_options(eod_df, "CALL",
                              (gld_price - 20, gld_price - 5), (25, 45))
    if len(itm_calls):
        best = itm_calls.iloc[len(itm_calls)//2]  # 取中间的
        # Find a good one with high OI
        high_oi = itm_calls.nlargest(3, "option_open_interest")
        for _, row in high_oi.iterrows():
            delta = abs(row.get("option_delta", 0))
            profit_if_up = (target_up - row["option_strike_price"]) - row["mid"]
            roi = profit_if_up / row["mid"] * 100 if row["mid"] > 0 else 0
            print(f"  │  → {_fmt_contract(row):20s}  中间价=${row['mid']:.2f}  "
                  f"Delta={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            print(f"  │    若涨至${target_up:.0f}: 盈利≈${profit_if_up:.0f}/张 "
                  f"(ROI≈{roi:+.0f}%)")
    else:
        print(f"  │  → 快照中无符合条件的合约 (需盘中实时报价)")
        print(f"  │  → 建议: Strike ${gld_price-15:.0f}~${gld_price-5:.0f}, "
              f"DTE 30-45天")
    print(f"  └{'─' * 54}")

    # ── 策略B: 中性 ──
    print(f"\n  ┌─ B. 中性策略: 买入 ATM Call ──────────────────────")
    print(f"  │  目标: Delta≈0.50, DTE 21-35天, 平衡杠杆与成本")
    print(f"  │  优点: 方向性收益好, 流动性最佳")
    print(f"  │  风险: 时间损耗中等, 需要较明确的方向判断")
    atm_calls = _find_options(eod_df, "CALL",
                              (gld_price - 5, gld_price + 5), (17, 35))
    if len(atm_calls):
        high_oi = atm_calls.nlargest(3, "option_open_interest")
        for _, row in high_oi.iterrows():
            delta = abs(row.get("option_delta", 0))
            move = target_up - gld_price
            # Per-share profit estimate: delta*move - theta*5days
            profit_per_share = delta * move - abs(row.get("option_theta", 0)) * 5
            roi = profit_per_share / row["mid"] * 100 if row["mid"] > 0 else 0
            print(f"  │  → {_fmt_contract(row):20s}  中间价=${row['mid']:.2f}  "
                  f"Delta={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            print(f"  │    若涨至${target_up:.0f}(5日): 估算盈利≈${profit_per_share:.1f}/股 "
                  f"(ROI≈{roi:+.0f}%)")
    else:
        print(f"  │  → 建议: Strike ${gld_price:.0f}, DTE 21-35天")
    print(f"  └{'─' * 54}")

    # ── 策略C: 激进 ──
    print(f"\n  ┌─ C. 激进策略: 买入 OTM Call ──────────────────────")
    print(f"  │  目标: Delta≈0.25-0.35, DTE 14-28天, 高杠杆")
    print(f"  │  优点: 成本低, 若大涨收益倍数高")
    print(f"  │  风险: 时间损耗快, 需要快速上涨才能盈利, 归零概率较高")
    otm_calls = _find_options(eod_df, "CALL",
                              (gld_price + 5, gld_price + 20), (14, 28))
    if len(otm_calls):
        high_oi = otm_calls.nlargest(3, "option_open_interest")
        for _, row in high_oi.iterrows():
            delta = abs(row.get("option_delta", 0))
            # OTM: if price reaches strike, value = intrinsic + time
            intrinsic_at_target = max(target_up - row["option_strike_price"], 0)
            profit = (intrinsic_at_target - row["mid"])
            roi = profit / row["mid"] * 100 if row["mid"] > 0 else 0
            print(f"  │  → {_fmt_contract(row):20s}  中间价=${row['mid']:.2f}  "
                  f"Delta={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            if intrinsic_at_target > 0:
                print(f"  │    若涨至${target_up:.0f}: 内在价值=${intrinsic_at_target:.0f} "
                      f"(ROI≈{roi:+.0f}%)")
            else:
                print(f"  │    需涨至>${row['option_strike_price']:.0f}才有内在价值")
    else:
        print(f"  │  → 建议: Strike ${gld_price+10:.0f}~${gld_price+20:.0f}, "
              f"DTE 14-28天")
    print(f"  └{'─' * 54}")


def _print_sell_put_strategies(gld_price, target_up, target_lo,
                                pred_u_pct, rv_pctile, eod_df):
    """SELL PUT 信号的三档策略."""
    print(f"\n  信号: SELL PUT (Bull + 低位 + 高波动率)")
    print(f"  策略核心: 高IV环境下卖出看跌期权收取高额权利金")

    # ── 策略A: 稳健 ──
    print(f"\n  ┌─ A. 稳健策略: 卖出深度OTM Put ────────────────────")
    print(f"  │  目标: Delta≈-0.10~-0.15, DTE 30-45天")
    print(f"  │  优点: 被行权概率极低, 占用保证金少")
    print(f"  │  风险: 收取权利金较少, 极端下跌时亏损")
    deep_otm_puts = _find_options(eod_df, "PUT",
                                   (gld_price * 0.88, gld_price * 0.93), (25, 45))
    if len(deep_otm_puts):
        for _, row in deep_otm_puts.nlargest(3, "option_open_interest").iterrows():
            delta = abs(row.get("option_delta", 0))
            premium = row["mid"]
            margin = row["option_strike_price"] * 0.15  # 粗估保证金
            ret_on_margin = premium / margin * 100
            print(f"  │  → {_fmt_contract(row):20s}  权利金=${premium:.2f}  "
                  f"|Delta|={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            print(f"  │    保证金≈${margin:.0f}, 5日收益率≈{ret_on_margin:.1f}%")
    else:
        print(f"  │  → 建议: Strike ${gld_price*0.90:.0f}~${gld_price*0.93:.0f}")
    print(f"  └{'─' * 54}")

    # ── 策略B: 中性 ──
    print(f"\n  ┌─ B. 中性策略: 卖出OTM Put ────────────────────────")
    print(f"  │  目标: Delta≈-0.20~-0.30, DTE 21-35天")
    print(f"  │  优点: 权利金适中, 有一定安全垫")
    otm_puts = _find_options(eod_df, "PUT",
                              (gld_price * 0.93, gld_price * 0.97), (17, 35))
    if len(otm_puts):
        for _, row in otm_puts.nlargest(3, "option_open_interest").iterrows():
            delta = abs(row.get("option_delta", 0))
            premium = row["mid"]
            breakeven = row["option_strike_price"] - premium
            print(f"  │  → {_fmt_contract(row):20s}  权利金=${premium:.2f}  "
                  f"|Delta|={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            print(f"  │    盈亏平衡=${breakeven:.0f} "
                  f"(距现价{(breakeven/gld_price-1)*100:.1f}%)")
    else:
        print(f"  │  → 建议: Strike ${gld_price*0.95:.0f}~${gld_price*0.97:.0f}")
    print(f"  └{'─' * 54}")

    # ── 策略C: 激进 ──
    print(f"\n  ┌─ C. 激进策略: 卖出ATM/近ATM Put ──────────────────")
    print(f"  │  目标: Delta≈-0.35~-0.45, DTE 14-28天")
    print(f"  │  优点: 权利金丰厚, 若不跌即获全部premium")
    print(f"  │  风险: 保证金高, 被行权风险较大")
    near_atm_puts = _find_options(eod_df, "PUT",
                                   (gld_price * 0.97, gld_price + 2), (14, 28))
    if len(near_atm_puts):
        for _, row in near_atm_puts.nlargest(3, "option_open_interest").iterrows():
            delta = abs(row.get("option_delta", 0))
            premium = row["mid"]
            breakeven = row["option_strike_price"] - premium
            print(f"  │  → {_fmt_contract(row):20s}  权利金=${premium:.2f}  "
                  f"|Delta|={delta:.2f}  OI={row['option_open_interest']:,.0f}")
            print(f"  │    盈亏平衡=${breakeven:.0f} "
                  f"(距现价{(breakeven/gld_price-1)*100:.1f}%)")
    else:
        print(f"  │  → 建议: Strike ${gld_price*0.98:.0f}~${gld_price:.0f}")
    print(f"  └{'─' * 54}")


def _print_exit_strategies(gld_price, eod_df):
    """EXIT 信号的操作建议."""
    print(f"\n  信号: EXIT (平仓)")
    print(f"\n  ┌─ 操作建议 ─────────────────────────────────────────")
    print(f"  │  1. 持有 Call 多头 → 市价平仓, 锁定利润")
    print(f"  │  2. 持有 Put 空头 → 如权利金已大幅衰减, 可等自然到期")
    print(f"  │     否则买入平仓, 避免反转风险")
    print(f"  │  3. 新开仓: 暂停, 等待下一个买入信号")
    print(f"  └{'─' * 54}")


def print_next_day_preview(close, range_df, upper_band, lower_band,
                            regime, rv_pctile, bp_dates):
    """预览下一个交易日的信号阈值."""
    today = bp_dates[-1]
    today_close_val = close.get(today, 0)

    # 计算下一交易日的 band
    # Upper = Daily lag1 = close(today) * (1 + pred_upper(today)/100)
    if today not in range_df.index:
        return
    pu_today = range_df.loc[today, "pred_upper_pct"]
    next_upper = today_close_val * (1 + pu_today / 100)

    # Lower = LagAvg(lag1,2,3) → 用 today, t-1, t-2 的预测
    lowers = []
    for lag_offset in range(3):
        idx = bp_dates.get_loc(today) - lag_offset
        if idx < 0:
            break
        d = bp_dates[idx]
        if d in range_df.index:
            c = close.get(d, 0)
            pl = range_df.loc[d, "pred_lower_pct"]
            lowers.append(c * (1 + pl / 100))
    next_lower = np.mean(lowers) if lowers else 0

    if next_upper <= next_lower:
        return

    next_bp030 = next_lower + 0.30 * (next_upper - next_lower)
    next_bp090 = next_lower + 0.90 * (next_upper - next_lower)

    # Regime and RV outlook
    today_regime = regime.get(today, "?")
    today_rv = rv_pctile.get(today, 0)

    print(f"\n  {'═' * 56}")
    print(f"  下一交易日信号阈值预览")
    print(f"  {'═' * 56}")
    print(f"  (基于 {today.date()} 收盘数据计算, Regime/RV沿用当日)")
    print(f"\n  Band 区间:")
    print(f"    上界: ${next_upper:.2f}")
    print(f"    下界: ${next_lower:.2f}")
    print(f"    宽度: ${next_upper - next_lower:.2f}")

    print(f"\n  信号触发价位:")
    if today_regime == "Bull":
        if today_rv <= 0.85:
            print(f"    BUY CALL: GLD < ${next_bp030:.2f} "
                  f"(Bull + RV {today_rv:.0%}≤85%)")
        else:
            print(f"    SELL PUT: GLD < ${next_bp030:.2f} "
                  f"(Bull + RV {today_rv:.0%}>85%)")
    else:
        print(f"    买入信号: 不触发 (当前Regime={today_regime}, 非Bull)")
    print(f"    EXIT:     GLD > ${next_bp090:.2f}")
    print(f"    无信号区: ${next_bp030:.2f} ~ ${next_bp090:.2f}")

    # bp at current close
    next_bp_at_close = (today_close_val - next_lower) / (next_upper - next_lower)
    print(f"\n  若开盘价≈${today_close_val:.2f} → bp={next_bp_at_close:.3f} "
          f"({'买入区' if next_bp_at_close < 0.30 else '观望区' if next_bp_at_close < 0.90 else '平仓区'})")


SIG_COLORS = {"BUY CALL": "#2196F3", "SELL PUT": "#FF9800"}
EXIT_MARKERS = {
    "BandExit": ("v", "#F44336"),
    "Pullback": ("s", "#FF6600"),
    "Timeout":  ("X", "gray"),
}


def build_trades(close, high, dates_all, buy_call, sell_put, exit_sig,
                 max_hold=10):
    """构建交易记录 (用于可视化)."""
    entries = []
    for d in dates_all:
        if buy_call.get(d, False):
            entries.append((d, "BUY CALL", close[d]))
        elif sell_put.get(d, False):
            entries.append((d, "SELL PUT", close[d]))

    all_dates = close.index
    trades = []
    for entry_date, sig_type, entry_price in entries:
        loc = all_dates.get_loc(entry_date)
        window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
        if len(window) == 0:
            continue
        exit_date, exit_type = None, "Timeout"
        peak = entry_price
        traj = [(entry_date, entry_price)]

        for fd in window:
            fc = close.get(fd, entry_price)
            fh = high.get(fd, fc)
            peak = max(peak, fh)
            traj.append((fd, fc))
            if exit_sig.get(fd, False):
                exit_date, exit_type = fd, "BandExit"
                break
            ppct = (peak / entry_price - 1) * 100
            dd = (peak - fc) / peak * 100
            if ppct > 2.0 and dd >= 1.5:
                exit_date, exit_type = fd, "Pullback"
                break

        if exit_date is None:
            exit_date, exit_type = window[-1], "Timeout"

        exit_price = close.get(exit_date, entry_price)
        g = (exit_price / entry_price - 1) * 100
        hd = all_dates.get_loc(exit_date) - loc
        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date,
            sig_type=sig_type, exit_type=exit_type,
            entry_price=entry_price, exit_price=exit_price,
            gain=g, hold_days=hd, trajectory=traj))
    return trades


def generate_dashboard(close, high, dates_all, upper_band, lower_band,
                       buy_call, sell_put, exit_sig, rv_pctile, regime,
                       range_df, today, today_close, pred_u_pct, pred_l_pct,
                       next_upper, next_lower, next_bp030, next_bp090,
                       eod_df, snap_date, signal_type, today_rv):
    """
    生成每日交易仪表板:
      上方: 近2个月K线 + Band + 信号 + 5日预测区间 + 下一日阈值
      下方: 期权策略推荐表格
    """
    trades = build_trades(close, high, dates_all, buy_call, sell_put, exit_sig)

    fig = plt.figure(figsize=(18, 14))
    # 上方图表占 65%, 下方表格占 35%
    gs = fig.add_gridspec(2, 1, height_ratios=[6.5, 3.5], hspace=0.08)
    ax = fig.add_subplot(gs[0])

    # ── 价格线 ──
    cl = close.loc[dates_all]
    ax.plot(dates_all, cl, "k-", lw=1.8, alpha=0.9, zorder=3)

    # ── Band ──
    ub = upper_band.reindex(dates_all).dropna()
    lb = lower_band.reindex(dates_all).dropna()
    ax.plot(ub.index, ub.values, color="green", lw=1, alpha=0.5)
    ax.plot(lb.index, lb.values, color="magenta", lw=1, alpha=0.5)
    cidx = ub.index.intersection(lb.index)
    ax.fill_between(cidx, lb.loc[cidx].values, ub.loc[cidx].values,
                     alpha=0.06, color="green")

    # ── Regime 背景 ──
    reg = regime.reindex(dates_all)
    bull = reg == "Bull"
    starts = dates_all[bull & (~bull.shift(1, fill_value=False))]
    ends = dates_all[bull & (~bull.shift(-1, fill_value=False))]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.04, color="green")

    # ── Exit 信号 ──
    entry_dates_set = set(t["entry_date"] for t in trades)
    ex_dates = [d for d in dates_all if exit_sig.get(d, False)
                and d not in entry_dates_set]
    if ex_dates:
        ax.scatter(ex_dates, [cl.get(d) for d in ex_dates],
                   marker="v", s=120, color="#F44336", edgecolors="darkred",
                   linewidths=0.7, zorder=5)
        for d in ex_dates:
            ax.annotate(f"{d.strftime('%m/%d')}", xy=(d, cl.get(d)),
                        xytext=(0, 12), textcoords="offset points",
                        fontsize=6, ha="center", color="#F44336",
                        fontweight="bold")

    # ── 交易轨迹 ──
    for t in trades:
        td = [x[0] for x in t["trajectory"]]
        tp = [x[1] for x in t["trajectory"]]
        c = SIG_COLORS[t["sig_type"]]
        ax.plot(td, tp, "-", color=c, lw=2,
                alpha=0.85 if t["gain"] > 0 else 0.4, zorder=4)
        ax.scatter([t["entry_date"]], [t["entry_price"]], marker="^",
                   s=160, color=c, edgecolors="black", linewidths=0.7,
                   zorder=6)
        rv_val = rv_pctile.get(t["entry_date"], np.nan)
        rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
        ax.annotate(f"{t['entry_date'].strftime('%m/%d')}{rv_txt}",
                    xy=(t["entry_date"], t["entry_price"]),
                    xytext=(0, -16), textcoords="offset points",
                    fontsize=7, ha="center", color=c, fontweight="bold")
        mk, mc = EXIT_MARKERS.get(t["exit_type"], ("o", "gray"))
        if t["exit_date"] not in entry_dates_set:
            ax.scatter([t["exit_date"]], [t["exit_price"]], marker=mk,
                       s=100, color=mc, edgecolors="black", linewidths=0.5,
                       zorder=7)
        oy = (16 if t["exit_date"] in entry_dates_set
              else (12 if t["gain"] > 0 else -14))
        ax.annotate(f"{t['gain']:+.1f}% ({t['hold_days']}d)",
                    xy=(t["exit_date"], t["exit_price"]),
                    xytext=(5, oy), textcoords="offset points",
                    fontsize=7, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              alpha=0.8, ec="none"))

    # ── RV 副轴 ──
    ax2 = ax.twinx()
    rv_plot = rv_pctile.reindex(dates_all).dropna()
    ax2.plot(rv_plot.index, rv_plot.values * 100, color="purple",
             lw=0.7, ls="--", alpha=0.3, zorder=1)
    ax2.axhline(85, color="purple", lw=0.5, ls=":", alpha=0.3)
    ax2.set_ylabel("RV%", fontsize=7, color="purple", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="purple", labelsize=6)
    for lab in ax2.get_yticklabels():
        lab.set_alpha(0.4)

    # ── 5日预测区间 (未来阴影) ──
    future_start = today + timedelta(days=1)
    future_end = today + timedelta(days=8)  # 覆盖5个交易日
    target_up = today_close * (1 + pred_u_pct / 100)
    target_lo = today_close * (1 + pred_l_pct / 100)
    ax.fill_between([future_start, future_end], [target_lo, target_lo],
                     [target_up, target_up],
                     alpha=0.12, color="gold", zorder=1)
    ax.plot([future_start, future_end], [target_up, target_up],
            color="goldenrod", lw=1.2, ls="--", alpha=0.7)
    ax.plot([future_start, future_end], [target_lo, target_lo],
            color="goldenrod", lw=1.2, ls="--", alpha=0.7)
    # 标注预测上下界
    ax.annotate(f"${target_up:.0f} (+{pred_u_pct:.1f}%)",
                xy=(future_end, target_up), fontsize=8, fontweight="bold",
                color="goldenrod", ha="right", va="bottom")
    ax.annotate(f"${target_lo:.0f} ({pred_l_pct:.1f}%)",
                xy=(future_end, target_lo), fontsize=8, fontweight="bold",
                color="goldenrod", ha="right", va="top")

    # ── 下一交易日信号阈值 (水平虚线) ──
    if next_bp030 > 0:
        ax.axhline(next_bp030, color="#2196F3", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        ax.annotate(f"BUY < ${next_bp030:.1f} (bp=0.30)",
                    xy=(dates_all[-1], next_bp030),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#2196F3", fontweight="bold",
                    ha="left", va="center")
    if next_bp090 > 0:
        ax.axhline(next_bp090, color="#F44336", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        ax.annotate(f"EXIT > ${next_bp090:.1f} (bp=0.90)",
                    xy=(dates_all[-1], next_bp090),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#F44336", fontweight="bold",
                    ha="left", va="center")

    # ── 当日标注 ──
    today_bp_val = (today_close - lower_band.get(today, 0)) / \
        (upper_band.get(today, 0) - lower_band.get(today, 0)) \
        if upper_band.get(today, 0) != lower_band.get(today, 0) else 0
    today_rv_val = rv_pctile.get(today, 0)
    marker_color = "#2196F3" if signal_type == "BUY_CALL" else \
        "#FF9800" if signal_type == "SELL_PUT" else \
        "#F44336" if signal_type == "EXIT" else "black"
    ax.scatter([today], [today_close], marker="D", s=120,
               color=marker_color, edgecolors="black", linewidths=1.5,
               zorder=8)
    sig_label_short = "BUY CALL" if signal_type == "BUY_CALL" else \
        "SELL PUT" if signal_type == "SELL_PUT" else \
        "EXIT" if signal_type == "EXIT" else ""
    ax.annotate(
        f"${today_close:.1f}  bp={today_bp_val:.2f}  RV={today_rv_val:.0%}"
        + (f"\n{sig_label_short}" if sig_label_short else ""),
        xy=(today, today_close), xytext=(-60, -28),
        textcoords="offset points", fontsize=7.5, fontweight="bold",
        ha="center", color=marker_color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=marker_color,
                  alpha=0.9))

    # ── 图表格式 ──
    sig_label = signal_type.replace("_", " ") if signal_type else "无信号"
    regime_now = regime.get(today, "?")
    ax.set_title(
        f"GLD 交易仪表板 — {today.strftime('%Y-%m-%d')}  |  "
        f"Regime: {regime_now}  |  "
        f"信号: {sig_label}  |  "
        f"GLD ${today_close:.2f}",
        fontsize=14, fontweight="bold")
    ax.set_ylabel("GLD ($)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # 每周一
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    # 留出右侧空间显示未来区间
    ax.set_xlim(dates_all[0] - timedelta(days=1),
                future_end + timedelta(days=2))

    # 图例
    legend_el = [
        Line2D([0], [0], color="black", lw=1.5, label="GLD"),
        Line2D([0], [0], color="green", lw=1, alpha=0.6, label="Upper Band"),
        Line2D([0], [0], color="magenta", lw=1, alpha=0.6, label="Lower Band"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2196F3",
               markersize=9, label="Buy Call"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF9800",
               markersize=9, label="Sell Put"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#F44336",
               markersize=9, label="Exit"),
        Line2D([0], [0], color="goldenrod", lw=1.2, ls="--",
               label="5d Range Pred"),
        Line2D([0], [0], color="#2196F3", lw=1.2, ls="-.",
               label="Buy Threshold"),
    ]
    ax.legend(handles=legend_el, loc="upper left", fontsize=7, ncol=4,
              framealpha=0.9)

    # ── 交易汇总 ──
    if trades:
        tdf = pd.DataFrame(trades)
        tg = tdf["gain"].mean()
        wr = (tdf["gain"] > 0).mean()
        hd = tdf["hold_days"].mean()
        n_bc = (tdf["sig_type"] == "BUY CALL").sum()
        n_sp = (tdf["sig_type"] == "SELL PUT").sum()
        summary = (f"BC({n_bc})+SP({n_sp})+Exit({len(ex_dates)}) | "
                   f"Avg:{tg:+.1f}% WR:{wr:.0%} Hold:{hd:.1f}d")
        ax.text(0.99, 0.02, summary, transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="right", va="bottom",
                bbox=dict(fc="lightyellow", ec="gray", alpha=0.85))

    # ══════════════════════════════════════════════════════
    # 下方: 期权策略推荐表格
    # ══════════════════════════════════════════════════════
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    _draw_options_table(ax_tbl, signal_type, today_close, pred_u_pct,
                        today_rv, eod_df, snap_date, next_bp030, next_bp090)

    # 保存
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    out_png = os.path.join(PROJECT_ROOT, "outputs", "daily_dashboard.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  仪表板已保存: {out_png}")
    return out_png


def _draw_options_table(ax, signal_type, gld_price, pred_u_pct,
                        rv_pctile_val, eod_df, snap_date,
                        next_bp030, next_bp090):
    """在 axes 上绘制期权策略推荐表格."""
    target_up = gld_price * (1 + pred_u_pct / 100)

    # 标题
    if signal_type == "BUY_CALL":
        title_text = "BUY CALL 期权策略推荐"
        title_color = "#2196F3"
    elif signal_type == "SELL_PUT":
        title_text = "SELL PUT 期权策略推荐"
        title_color = "#FF9800"
    elif signal_type == "EXIT":
        title_text = "EXIT 平仓建议"
        title_color = "#F44336"
    else:
        title_text = "当前无信号 — 观望"
        title_color = "gray"

    ax.text(0.5, 0.97, title_text, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="top",
            color=title_color)

    # 下一日阈值信息
    ax.text(0.5, 0.90,
            f"下一交易日:  买入 < ${next_bp030:.1f}   |   "
            f"平仓 > ${next_bp090:.1f}   |   "
            f"5日目标 ${target_up:.0f} (+{pred_u_pct:.1f}%)",
            transform=ax.transAxes, fontsize=10, ha="center", va="top",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="gray",
                      alpha=0.8))

    if eod_df is None or signal_type not in ("BUY_CALL", "SELL_PUT"):
        if signal_type == "EXIT":
            ax.text(0.5, 0.55,
                    "持有 Call → 市价平仓锁定利润\n"
                    "持有 Put空头 → 权利金衰减后可等到期, 否则买入平仓\n"
                    "新开仓 → 暂停, 等待下一个买入信号",
                    transform=ax.transAxes, fontsize=11, ha="center",
                    va="center", linespacing=1.8)
        elif signal_type is None:
            ax.text(0.5, 0.55,
                    "当前 bp 在 0.30~0.90 之间, 无买入/平仓信号\n"
                    "持仓中请关注止盈拐点 (Pullback / MACD / RSI)",
                    transform=ax.transAxes, fontsize=11, ha="center",
                    va="center", linespacing=1.8)
        return

    # 构建策略表格数据
    rows = []
    if signal_type == "BUY_CALL":
        configs = [
            ("A. 稳健 (ITM Call)", "CALL",
             (gld_price - 20, gld_price - 5), (25, 45),
             "Delta~0.70, 跟踪紧密", "#4CAF50"),
            ("B. 中性 (ATM Call)", "CALL",
             (gld_price - 5, gld_price + 5), (17, 35),
             "Delta~0.50, 平衡杠杆", "#2196F3"),
            ("C. 激进 (OTM Call)", "CALL",
             (gld_price + 5, gld_price + 20), (14, 28),
             "Delta~0.30, 高杠杆", "#FF5722"),
        ]
    else:  # SELL_PUT
        configs = [
            ("A. 稳健 (深OTM Put)", "PUT",
             (gld_price * 0.88, gld_price * 0.93), (25, 45),
             "|Delta|~0.10, 极低行权概率", "#4CAF50"),
            ("B. 中性 (OTM Put)", "PUT",
             (gld_price * 0.93, gld_price * 0.97), (17, 35),
             "|Delta|~0.25, 适中权利金", "#2196F3"),
            ("C. 激进 (近ATM Put)", "PUT",
             (gld_price * 0.97, gld_price + 2), (14, 28),
             "|Delta|~0.40, 高权利金", "#FF5722"),
        ]

    for label, otype, strike_range, dte_range, desc, color in configs:
        opts = _find_options(eod_df, otype, strike_range, dte_range)
        if len(opts):
            best = opts.nlargest(1, "option_open_interest").iloc[0]
            exp = pd.Timestamp(best["strike_time"]).strftime("%m/%d")
            strike = best["option_strike_price"]
            delta = abs(best.get("option_delta", 0))
            mid = best["mid"]
            oi = best["option_open_interest"]
            iv = best.get("iv_decimal", 0) * 100

            if otype == "CALL":
                intrinsic = max(target_up - strike, 0)
                roi = (intrinsic - mid) / mid * 100 if mid > 0 else 0
                roi_str = f"{roi:+.0f}%"
            else:
                # Sell put: max profit = premium
                roi_str = f"+{mid/strike*100:.1f}%"

            rows.append([
                label, f"GLD {exp} ${strike:.0f}{'C' if otype=='CALL' else 'P'}",
                f"${mid:.2f}", f"{delta:.2f}", f"{iv:.0f}%",
                f"{oi:,.0f}", roi_str, desc, color
            ])
        else:
            rows.append([
                label, "—", "—", "—", "—", "—", "—", desc, color
            ])

    if not rows:
        return

    # 绘制表格
    col_labels = ["策略", "合约", "中间价", "Delta", "IV", "OI", "目标ROI", "说明"]
    col_widths = [0.12, 0.16, 0.08, 0.07, 0.06, 0.09, 0.08, 0.26]
    n_cols = len(col_labels)
    n_rows = len(rows)

    y_top = 0.78
    row_h = 0.11
    x_start = 0.04

    # 表头
    x = x_start
    for j, (label, w) in enumerate(zip(col_labels, col_widths)):
        ax.text(x + w / 2, y_top, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="center", va="center",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="#555555", ec="none"))
        x += w

    # 数据行
    for i, row in enumerate(rows):
        y = y_top - (i + 1) * row_h
        row_color = row[-1]
        bg_color = row_color + "15"  # 半透明

        x = x_start
        for j, (val, w) in enumerate(zip(row[:-1], col_widths)):
            weight = "bold" if j == 0 else "normal"
            fc = row_color if j == 0 else "black"
            fontsize = 9 if j in (0, 1, 7) else 10
            ax.text(x + w / 2, y, val, transform=ax.transAxes,
                    fontsize=fontsize, fontweight=weight,
                    ha="center", va="center", color=fc)
            x += w

        # 行背景
        rect = FancyBboxPatch(
            (x_start, y - row_h * 0.4), sum(col_widths), row_h * 0.8,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01", fc=row_color, alpha=0.06,
            ec=row_color, linewidth=0.5)
        ax.add_patch(rect)

    # 风控提示
    ax.text(0.5, y_top - (n_rows + 1) * row_h - 0.02,
            f"风控: 仓位 2-5%(稳健)/5-10%(中性)/≤5%(激进)  |  "
            f"平仓: bp>0.90 / Pullback(涨>2%回撤≥1.5%) / 10d timeout  |  "
            f"快照: {snap_date}  |  覆盖率: 74%",
            transform=ax.transAxes, fontsize=8, ha="center", va="top",
            color="gray", style="italic")


def main():
    print("=" * 60)
    print("  GLD 区间预测 & 信号生成")
    print("=" * 60)

    # ── 1. 加载数据 ──
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    common = features.index.intersection(gld.index)
    features = features.loc[common]
    gld = gld.loc[common]

    close = gld["Close"]
    print(f"\n  数据范围: {common[0].date()} ~ {common[-1].date()} ({len(common)} 天)")
    print(f"  GLD 最新: ${close.iloc[-1]:.2f} ({common[-1].date()})")

    # ── 2. 构建目标 ──
    high, low = gld["High"], gld["Low"]
    max_high_5d = high.shift(-1).rolling(5).max().shift(-4)
    min_low_5d = low.shift(-1).rolling(5).min().shift(-4)
    upper_pct = (max_high_5d / close - 1) * 100
    lower_pct = (min_low_5d / close - 1) * 100
    log_ret = np.log(close / close.shift(1))
    rv_scale = log_ret.rolling(10).std() * np.sqrt(5) * 100

    feat_cols = select_features(features)
    valid = features[feat_cols].dropna().index
    valid = valid.intersection(rv_scale.dropna().index)
    dates = valid

    upper_s = upper_pct.reindex(dates)
    lower_s = lower_pct.reindex(dates)
    rv_s = rv_scale.reindex(dates)

    # ── 3. 训练最后一折 ──
    # 用前面大部分数据训练, 最后126天做cal, 预测最近的日期
    n = len(dates)
    seq_len = 20
    cal_size = 126
    val_size = 252

    # 加载已有 OOS 预测
    oos_path = os.path.join(PROJECT_ROOT, "data", "models",
                            "dl_range_v2_oos.parquet")
    existing_oos = pd.read_parquet(oos_path)
    last_oos_date = existing_oos.index.max()

    # 找到需要预测的新日期
    new_dates_mask = dates > last_oos_date
    new_dates = dates[new_dates_mask]

    if len(new_dates) == 0:
        print(f"  OOS 预测已覆盖到 {last_oos_date.date()}, 无需新预测")
        range_df = existing_oos
    else:
        print(f"\n  已有 OOS 到: {last_oos_date.date()}")
        print(f"  需要预测: {new_dates[0].date()} ~ {new_dates[-1].date()} "
              f"({len(new_dates)} 天)")

        # 训练数据: 到 last_oos_date - cal_size
        cutoff_idx = dates.get_loc(last_oos_date) + 1
        # 确保至少有 cal_size 的校准数据
        train_end = cutoff_idx - cal_size
        cal_start = cutoff_idx - cal_size

        train_dates = dates[:train_end]
        val_start = max(0, train_end - val_size)
        val_dates = dates[val_start:train_end]
        cal_dates = dates[cal_start:cutoff_idx]

        # 预测区间: 从 cutoff 到最后
        test_dates = dates[cutoff_idx:]

        print(f"  Train: {train_dates[0].date()} ~ {train_dates[-1].date()} "
              f"({len(train_dates)} 天)")
        print(f"  Cal:   {cal_dates[0].date()} ~ {cal_dates[-1].date()} "
              f"({len(cal_dates)} 天)")
        print(f"  Test:  {test_dates[0].date()} ~ {test_dates[-1].date()} "
              f"({len(test_dates)} 天)")

        X_train = features.loc[train_dates, feat_cols].values
        u_train = upper_s.loc[train_dates].values
        l_train = lower_s.loc[train_dates].values
        rv_train = rv_s.loc[train_dates].values

        X_val = features.loc[val_dates, feat_cols].values
        u_val = upper_s.loc[val_dates].values
        l_val = lower_s.loc[val_dates].values
        rv_val = rv_s.loc[val_dates].values

        X_cal = features.loc[cal_dates, feat_cols].values
        u_cal = upper_s.loc[cal_dates].values
        l_cal = lower_s.loc[cal_dates].values
        rv_cal = rv_s.loc[cal_dates].values

        print("\n  训练 DL Range 模型 (3种子集成)...")
        predictor = DLRangePredictor(
            seq_len=seq_len, hidden_size=64, num_layers=2,
            dropout=0.2, lr=1e-3, weight_decay=1e-4,
            epochs=150, batch_size=64, patience=20,
            q_upper=0.85, q_lower=0.15,
            n_ensemble=3, cal_target_cov=0.80,
        )
        predictor.fit(
            X_train, u_train, l_train, rv_train,
            X_val, u_val, l_val, rv_val,
            X_cal, u_cal, l_cal, rv_cal,
            verbose=False,
        )
        print(f"  Conformal margins: upper=+{predictor.cal_upper_margin:.2f}%, "
              f"lower=+{predictor.cal_lower_margin:.2f}%")

        # 预测
        n_prefix = seq_len - 1
        prefix_dates = dates[cutoff_idx - n_prefix: cutoff_idx]
        combined_dates = prefix_dates.append(test_dates)
        X_comb = features.loc[combined_dates, feat_cols].values
        rv_comb = rv_s.loc[combined_dates].values

        pred_u, pred_l = predictor.predict(X_comb, rv_comb)
        assert len(pred_u) == len(test_dates), \
            f"pred len {len(pred_u)} != test len {len(test_dates)}"

        # 合并到已有 OOS
        new_oos = pd.DataFrame({
            "pred_upper_pct": pred_u,
            "pred_lower_pct": pred_l,
            "actual_upper_pct": upper_s.loc[test_dates].values,
            "actual_lower_pct": lower_s.loc[test_dates].values,
            "gld_close": close.loc[test_dates].values,
        }, index=test_dates)

        range_df = pd.concat([existing_oos, new_oos])
        range_df = range_df[~range_df.index.duplicated(keep="last")]
        range_df.to_parquet(oos_path)
        print(f"  OOS 预测已更新: {range_df.index[-1].date()}")

    # ── 4. 计算 Hybrid Band + 信号 ──
    feat_cols_r = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols_r])["regime"]
    rv_10d = features["rv_10d"] if "rv_10d" in features.columns else None
    rv_pctile = compute_rv_pctile(rv_10d)

    upper_band, lower_band, bp = build_band(
        range_df, close,
        upper_lags=(1,), lower_lags=(1, 2, 3))

    bp_dates = bp.dropna().index
    bp_s = bp.reindex(bp_dates)
    rv_p = rv_pctile.reindex(bp_dates)
    is_bull = (regime.reindex(bp_dates) == "Bull")
    buy_call, sell_put, exit_sig = generate_v2_signals(bp_s, rv_p, is_bull)

    # ── 5. 输出今日结果 ──
    today = bp_dates[-1]
    today_close = close.get(today, 0)
    today_bp = bp_s.get(today, 0)
    today_regime = regime.get(today, "?")
    today_rv = rv_p.get(today, 0)
    today_ub = upper_band.get(today, 0)
    today_lb = lower_band.get(today, 0)

    # 信号
    sig = "无"
    if buy_call.get(today, False):
        sig = "BUY CALL (买入看涨期权)"
    elif sell_put.get(today, False):
        sig = "SELL PUT (卖出看跌期权)"
    if exit_sig.get(today, False):
        sig += " + EXIT (平仓)" if sig != "无" else "EXIT (平仓)"

    # 最新预测区间
    last_pred_date = range_df.index[-1]
    pred_u = range_df.loc[last_pred_date, "pred_upper_pct"]
    pred_l = range_df.loc[last_pred_date, "pred_lower_pct"]
    pred_close = close.get(last_pred_date, today_close)

    # 5天预测区间 (基于最新预测)
    upper_price = pred_close * (1 + pred_u / 100)
    lower_price = pred_close * (1 + pred_l / 100)

    print(f"\n{'=' * 60}")
    print(f"  预测日期: {today.date()} (数据截至)")
    print(f"{'=' * 60}")
    print(f"\n  GLD 收盘: ${today_close:.2f}")
    print(f"  Regime:   {today_regime}")
    print(f"  RV 分位:  {today_rv:.1%}")
    print(f"  Band位置: {today_bp:.2f}")
    print(f"    上界:   ${today_ub:.2f}")
    print(f"    下界:   ${today_lb:.2f}")

    print(f"\n  ═══ 信号: {sig} ═══")

    print(f"\n  5天区间预测 (基于 {last_pred_date.date()}):")
    print(f"    上界: ${upper_price:.2f} (+{pred_u:.1f}%)")
    print(f"    下界: ${lower_price:.2f} ({pred_l:.1f}%)")

    # ── 6. 多品种价格区间 ──
    # 模型输出是百分比区间, 直接应用到各品种当前价格
    # 用户需代入实时行情
    up_pct = pred_u
    lo_pct = pred_l

    print(f"\n  ═══ 预测区间 (百分比) ═══")
    print(f"    上界: +{up_pct:.1f}%")
    print(f"    下界: {lo_pct:.1f}%")
    print(f"\n  将上述百分比应用到各品种当前价格即可:")
    print(f"    品种当前价 × (1 + {up_pct:.1f}%) = 5日上界")
    print(f"    品种当前价 × (1 + ({lo_pct:.1f}%)) = 5日下界")

    # 用 COMEX 金价反推换算 (如果有数据)
    gold_futures_path = os.path.join(raw_dir, "market", "gold_futures.csv")
    if os.path.exists(gold_futures_path):
        gold_f = pd.read_csv(gold_futures_path, index_col=0, parse_dates=True)
        gc_close = gold_f["Close"].reindex(close.index).ffill()
        gc_latest = gc_close.iloc[-1]
        gc_date = gc_close.index[-1]
        gld_gc_ratio = gc_latest / close.iloc[-1]
    else:
        gc_latest = today_close * 10.9  # 近似
        gld_gc_ratio = 10.9

    print(f"\n  ═══ 多品种参考 (基于模型收盘数据) ═══")
    print(f"\n  {'品种':<18} {'模型价格':>12} {'5日上界':>12} {'5日下界':>12}")
    print(f"  {'-'*56}")

    gld_up = today_close * (1 + up_pct / 100)
    gld_lo = today_close * (1 + lo_pct / 100)
    gc_up = gc_latest * (1 + up_pct / 100)
    gc_lo = gc_latest * (1 + lo_pct / 100)

    print(f"  {'GLD ETF':<18} {'$'+f'{today_close:.2f}':>12} "
          f"{'$'+f'{gld_up:.2f}':>12} {'$'+f'{gld_lo:.2f}':>12}")
    print(f"  {'伦敦/纽约金 ($/oz)':<18} {'$'+f'{gc_latest:.0f}':>12} "
          f"{'$'+f'{gc_up:.0f}':>12} {'$'+f'{gc_lo:.0f}':>12}")

    print(f"\n  注: 请以实际行情价格代入 +{up_pct:.1f}% / {lo_pct:.1f}% 计算精确区间")
    print(f"       沪金需额外乘以 (USDCNY / 31.1035) 换算为 CNY/g")

    # 最近几天的信号历史
    print(f"\n  ═══ 近期信号 ═══")
    for d in bp_dates[-10:]:
        s = ""
        if buy_call.get(d, False):
            s = "BUY CALL"
        elif sell_put.get(d, False):
            s = "SELL PUT"
        if exit_sig.get(d, False):
            s += " EXIT" if s else "EXIT"
        r = regime.get(d, "?")
        b = bp_s.get(d, 0)
        c = close.get(d, 0)
        if s:
            print(f"  {d.date()} GLD=${c:.1f} bp={b:.2f} "
                  f"regime={r} → {s}")
        else:
            print(f"  {d.date()} GLD=${c:.1f} bp={b:.2f} regime={r}")

    # ── 7. 期权策略建议 (触发信号时) ──
    eod_df, snap_date = load_latest_eod_snapshot()

    if buy_call.get(today, False):
        print_strategy_recommendations(
            "BUY_CALL", today_close, pred_u, pred_l,
            today_rv, eod_df, snap_date)
    elif sell_put.get(today, False):
        print_strategy_recommendations(
            "SELL_PUT", today_close, pred_u, pred_l,
            today_rv, eod_df, snap_date)
    if exit_sig.get(today, False):
        print_strategy_recommendations(
            "EXIT", today_close, pred_u, pred_l,
            today_rv, eod_df, snap_date)

    # ── 8. 下一交易日信号阈值预览 ──
    print_next_day_preview(close, range_df, upper_band, lower_band,
                           regime, rv_pctile, bp_dates)

    # ── 9. 生成可视化仪表板 ──
    # 计算下一日 band
    next_upper, next_lower = 0, 0
    next_bp030, next_bp090 = 0, 0
    if today in range_df.index:
        pu_today = range_df.loc[today, "pred_upper_pct"]
        next_upper = today_close * (1 + pu_today / 100)
        lowers_next = []
        for lag_offset in range(3):
            idx = bp_dates.get_loc(today) - lag_offset
            if idx < 0:
                break
            d = bp_dates[idx]
            if d in range_df.index:
                c = close.get(d, 0)
                pl = range_df.loc[d, "pred_lower_pct"]
                lowers_next.append(c * (1 + pl / 100))
        next_lower = np.mean(lowers_next) if lowers_next else 0
        if next_upper > next_lower:
            next_bp030 = next_lower + 0.30 * (next_upper - next_lower)
            next_bp090 = next_lower + 0.90 * (next_upper - next_lower)

    # 近2个月的日期范围
    lookback = today - timedelta(days=65)
    viz_dates = close.index[(close.index >= lookback) & (close.index <= today)]

    # 确定当日信号类型
    sig_type_viz = None
    if buy_call.get(today, False):
        sig_type_viz = "BUY_CALL"
    elif sell_put.get(today, False):
        sig_type_viz = "SELL_PUT"
    if exit_sig.get(today, False):
        sig_type_viz = sig_type_viz or "EXIT"

    generate_dashboard(
        close, high, viz_dates, upper_band, lower_band,
        buy_call, sell_put, exit_sig, rv_pctile, regime,
        range_df, today, today_close, pred_u, pred_l,
        next_upper, next_lower, next_bp030, next_bp090,
        eod_df, snap_date, sig_type_viz, today_rv)


if __name__ == "__main__":
    main()
