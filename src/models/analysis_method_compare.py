"""
方法对比可视化: 基线 A vs 优化 V2 (期权视角)

信号类型:
  Buy Call:  Bull + bp<0.30 + RV≤85%  (正常波动率, 买call)
  Sell Put:  Bull + bp<0.30 + RV>85%  (高IV, 卖put收premium)
  Sell/TP:   bp>0.90 ∪ RV进入>85% ∪ Regime退出Bull

触发方式: 水平触发 (每天判定, 非穿越)

用法:
    conda activate gold
    python src/models/analysis_method_compare.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "regime_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_all():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]
    rv_10d = features["rv_10d"] if "rv_10d" in features.columns else None
    return gld, range_df, regime, rv_10d


def build_band(range_df, gld_close, lags=(1, 2, 3),
               upper_lags=None, lower_lags=None):
    """构建价格区间. 支持上下界不同lags.
    upper_lags/lower_lags: 若指定则上下界分别用不同lags, 否则都用lags.
    """
    close = gld_close.reindex(range_df.index)
    u_lags = upper_lags if upper_lags is not None else lags
    l_lags = lower_lags if lower_lags is not None else lags
    uppers = []
    for lag in u_lags:
        cl = close.shift(lag)
        pu = range_df["pred_upper_pct"].shift(lag)
        uppers.append(cl * (1 + pu / 100))
    lowers = []
    for lag in l_lags:
        cl = close.shift(lag)
        pl = range_df["pred_lower_pct"].shift(lag)
        lowers.append(cl * (1 + pl / 100))
    upper_band = pd.concat(uppers, axis=1).mean(axis=1)
    lower_band = pd.concat(lowers, axis=1).mean(axis=1)
    bp = (close - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bp


def compute_rv_pctile(rv, window=252):
    """RV百分位排名. 输入rv_10d或rv_20d均可."""
    return rv.rolling(window, min_periods=60).rank(pct=True)


# ==========================================================
# V2 信号生成 (水平触发 + 期权类型区分)
# ==========================================================
def generate_v2_signals(bp_s, rv_p, is_bull):
    """
    返回:
      buy_call: Bull + bp<0.30 + RV≤85%  → 买call
      sell_put: Bull + bp<0.30 + RV>85%  → 卖put (高IV, 收premium)
      sell:     bp>0.90 ∪ Regime退出Bull  → 止盈/平仓

    触发方式:
      买入区 (buy_call/sell_put): 水平触发 (每天判定)
      卖出 (sell): 水平触发 (bp>0.90每天都是卖出区) + 事件触发 (Regime退出)
      RV>85% 不再作为独立sell信号, 而是改变买入的类型 (call→put)
    """
    rv_high = rv_p > 0.85

    # 买入区: Bull + bp<0.30 (水平触发)
    buy_zone = is_bull & (bp_s < 0.30)
    buy_call = buy_zone & (~rv_high)       # 正常RV → buy call
    sell_put = buy_zone & rv_high          # 高RV → sell put (利用高IV)

    # 卖出区: bp>0.90 (水平) ∪ Regime退出(事件)
    bull_exit = is_bull.shift(1).fillna(False) & (~is_bull)
    sell = (bp_s > 0.90) | bull_exit

    return buy_call, sell_put, sell


def generate_baseline_signals(bp_s, is_bull):
    """方法A: 穿越触发 (保持原逻辑作为对照)"""
    bp_down_020 = (bp_s < 0.20) & (bp_s.shift(1) >= 0.20)
    bp_up_080 = (bp_s > 0.80) & (bp_s.shift(1) <= 0.80)
    buy = (is_bull & bp_down_020)
    sell = bp_up_080
    return buy, sell


# ==========================================================
# 止盈拐点检测
# ==========================================================
def compute_tp_indicators(close, high, low):
    """计算止盈所需的技术指标."""
    # MACD (fast=8, slow=17, signal=6) — 偏短线
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=6, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # RSI(7) — 短周期
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=7, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=7, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    return {
        "macd_line": macd_line, "macd_signal": macd_signal,
        "macd_hist": macd_hist, "rsi": rsi,
    }


def find_tp_exits(buy_dates, close, high, tp_ind, max_hold=10):
    """为每个买入信号找止盈退出点.

    每天同时检查5个条件, 最早触发的即退出 (同天触发则都记录, 取第一个):
      MACD:     MACD histogram 由正转负 (动量拐点, 要求已盈利>0.3%)
      MACDweak: MACD histogram 连续2天缩小 (动量减弱, 要求涨幅>1%)
      RSI:      RSI(7) 从超买回落 (>70 → <60, 要求已盈利)
      Pullback: 从peak回落 ≥1.5%, 要求peak已涨>2% (保护大幅获利)
      Timeout:  持仓到 max_hold 天 (兜底)

    返回: list of dict {entry, exit, exit_type, gain_pct, peak_pct}
    """
    macd_hist = tp_ind["macd_hist"]
    rsi = tp_ind["rsi"]
    all_dates = close.index

    trades = []
    for entry_date in buy_dates:
        if entry_date not in close.index:
            continue
        entry_price = close.loc[entry_date]
        entry_loc = all_dates.get_loc(entry_date)

        end_loc = min(entry_loc + max_hold + 1, len(all_dates))
        window = all_dates[entry_loc + 1: end_loc]
        if len(window) == 0:
            continue

        peak_price = entry_price
        exit_date = None
        exit_type = None

        for d in window:
            cur_high = high.get(d, np.nan)
            cur_close = close.get(d, np.nan)
            if np.isnan(cur_close):
                continue

            peak_price = max(peak_price, cur_high if not np.isnan(cur_high) else cur_close)
            cur_gain = (cur_close / entry_price - 1) * 100
            peak_gain = (peak_price / entry_price - 1) * 100
            d_loc = all_dates.get_loc(d)
            prev_d = all_dates[d_loc - 1] if d_loc > 0 else None
            prev2_d = all_dates[d_loc - 2] if d_loc > 1 else None

            mh_cur = macd_hist.get(d, 0)
            mh_prev = macd_hist.get(prev_d, 0) if prev_d else 0
            mh_prev2 = macd_hist.get(prev2_d, 0) if prev2_d else 0
            rsi_cur = rsi.get(d, 50)
            rsi_prev = rsi.get(prev_d, 50) if prev_d else 50

            # 同时检查所有条件, 收集当天触发的
            triggered = []

            # MACD hist 由正转负 (动量拐点)
            if mh_prev >= 0 and mh_cur < 0 and cur_gain > 0.3:
                triggered.append("MACD")

            # MACD hist 连续2天缩小 (动量衰减)
            if (cur_gain > 1.0 and mh_cur > 0
                    and mh_cur < mh_prev and mh_prev < mh_prev2
                    and mh_prev2 > 0):
                triggered.append("MACDweak")

            # RSI 超买回落
            if rsi_prev > 70 and rsi_cur < 60 and cur_gain > 0:
                triggered.append("RSI")

            # 从peak回落 (保护大幅获利)
            if peak_gain > 2.0:
                drawdown = (peak_price - cur_close) / peak_price * 100
                if drawdown >= 1.5:
                    triggered.append("Pullback")

            if triggered:
                exit_date = d
                exit_type = triggered[0]  # 同天多个触发取第一个
                break

        # 兜底: 超时退出
        if exit_date is None and len(window) > 0:
            exit_date = window[-1]
            exit_type = "Timeout"

        if exit_date is not None:
            exit_price = close.get(exit_date, np.nan)
            gain = (exit_price / entry_price - 1) * 100 if not np.isnan(exit_price) else 0
            pk = (peak_price / entry_price - 1) * 100
            hold_days = (all_dates.get_loc(exit_date) - entry_loc)
            trades.append({
                "entry": entry_date, "exit": exit_date,
                "exit_type": exit_type, "hold_days": hold_days,
                "gain_pct": gain, "peak_pct": pk,
                "entry_price": entry_price,
                "exit_price": exit_price if not np.isnan(exit_price) else entry_price,
            })

    return trades


# ==========================================================
# 评估
# ==========================================================
def eval_signals(buy_mask, sell_mask, close, high=None):
    results = {}

    buy_idx = buy_mask[buy_mask].index
    buy_idx = buy_idx[buy_idx.isin(close.index)]
    results["buy_n"] = len(buy_idx)

    for h in [1, 2, 3, 5, 10]:
        fwd = ((close.shift(-h) / close - 1) * 100).reindex(buy_idx).dropna()
        if len(fwd) > 0:
            results[f"buy_wr_{h}"] = (fwd > 0).mean()
            results[f"buy_avg_{h}"] = fwd.mean()

    if high is not None:
        for window in [1, 2, 3, 5, 10]:
            max_high = high.rolling(window).max().shift(-window)
            max_gain = ((max_high / close - 1) * 100).reindex(buy_idx).dropna()
            if len(max_gain) > 0:
                for tp in [1.0, 2.0, 3.0]:
                    results[f"buy_tp{tp:.0f}_{window}d"] = (max_gain >= tp).mean()
                results[f"buy_maxgain_{window}d"] = max_gain.mean()

    sell_idx = sell_mask[sell_mask].index
    sell_idx = sell_idx[sell_idx.isin(close.index)]
    results["sell_n"] = len(sell_idx)

    for h in [1, 2, 3, 5, 10]:
        fwd = ((close.shift(-h) / close - 1) * 100).reindex(sell_idx).dropna()
        if len(fwd) > 0:
            results[f"sell_nr1_{h}"] = (fwd < 1.0).mean()
            results[f"sell_nr2_{h}"] = (fwd < 2.0).mean()
            results[f"sell_avg_{h}"] = fwd.mean()

    # 分布数据
    for prefix, idx in [("buy", buy_idx), ("sell", sell_idx)]:
        for h in [5, 10]:
            fwd = ((close.shift(-h) / close - 1) * 100).reindex(idx).dropna()
            results[f"{prefix}_fwd_{h}d"] = fwd

    return results


# ==========================================================
# 可视化
# ==========================================================
def plot_v2(gld, regime, upper_band, lower_band, bp, rv_pctile,
            buy_call, sell_put, sell, stats_all, method_name, filename,
            date_start=None, date_end=None):
    """V2 专用可视化: 区分 Buy Call / Sell Put / Sell"""

    close = gld["Close"]
    reg = regime.reindex(close.index)

    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[3, 1.5, 1.5],
                  hspace=0.3, wspace=0.3)

    # ============================================================
    # Panel 1: 价格 + 多类型信号 (跨3列)
    # ============================================================
    ax = fig.add_subplot(gs[0, :])

    common = close.index[close.index.isin(bp.dropna().index)]
    if date_start is not None:
        common = common[common >= pd.Timestamp(date_start)]
    if date_end is not None:
        common = common[common <= pd.Timestamp(date_end)]
    cl = close.reindex(common)
    ub = upper_band.reindex(common)
    lb = lower_band.reindex(common)

    ax.plot(common, cl, color="black", linewidth=1, label="GLD Close", zorder=3)
    ax.fill_between(common, lb, ub, alpha=0.08, color="steelblue",
                    label="DL Range Band")

    # Regime shading
    reg_c = reg.reindex(common)
    bull = reg_c == "Bull"
    starts = common[bull & (~bull.shift(1, fill_value=False))]
    ends = common[bull & (~bull.shift(-1, fill_value=False))]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.06, color="green")

    is_zoomed = date_start is not None
    mk_size = 200 if is_zoomed else 100

    rv_s = rv_pctile.reindex(common)

    # --- Buy Call ---
    bc_idx = buy_call[buy_call].index.intersection(common)
    if len(bc_idx) > 0:
        ax.scatter(bc_idx, cl.reindex(bc_idx), marker="^", s=mk_size,
                   color="#2196F3", edgecolors="darkblue", linewidths=0.8,
                   zorder=5, label=f"Buy Call (n={len(bc_idx)})")
        if is_zoomed:
            for d in bc_idx:
                rv_val = rv_s.get(d, np.nan)
                rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
                ax.annotate(f"{d.strftime('%m/%d')} ${cl.loc[d]:.0f}{rv_txt}",
                            xy=(d, cl.loc[d]), xytext=(0, -22),
                            textcoords="offset points", fontsize=7,
                            ha="center", color="#2196F3", fontweight="bold")

    # --- Sell Put ---
    sp_idx = sell_put[sell_put].index.intersection(common)
    if len(sp_idx) > 0:
        ax.scatter(sp_idx, cl.reindex(sp_idx), marker="^", s=mk_size,
                   color="#FF9800", edgecolors="#E65100", linewidths=0.8,
                   zorder=5, label=f"Sell Put (n={len(sp_idx)})")
        if is_zoomed:
            for d in sp_idx:
                rv_val = rv_s.get(d, np.nan)
                rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
                ax.annotate(f"{d.strftime('%m/%d')} ${cl.loc[d]:.0f}{rv_txt}",
                            xy=(d, cl.loc[d]), xytext=(0, -22),
                            textcoords="offset points", fontsize=7,
                            ha="center", color="#FF9800", fontweight="bold")

    # --- 平仓 ---
    sl_idx = sell[sell].index.intersection(common)
    if len(sl_idx) > 0:
        ax.scatter(sl_idx, cl.reindex(sl_idx), marker="v", s=mk_size,
                   color="#F44336", edgecolors="darkred", linewidths=0.8,
                   zorder=5, label=f"Exit (n={len(sl_idx)})")
        if is_zoomed:
            for d in sl_idx:
                rv_val = rv_s.get(d, np.nan)
                rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
                ax.annotate(f"{d.strftime('%m/%d')} ${cl.loc[d]:.0f}{rv_txt}",
                            xy=(d, cl.loc[d]), xytext=(0, 14),
                            textcoords="offset points", fontsize=7,
                            ha="center", color="#F44336", fontweight="bold")

    ax.set_ylabel("GLD Price ($)", fontsize=12)
    ax.set_title(f"{method_name}", fontsize=14, fontweight="bold")

    # --- RV percentile 副轴 ---
    ax2 = ax.twinx()
    rv_plot = rv_s.dropna()
    ax2.plot(rv_plot.index, rv_plot.values * 100, color="purple",
             linewidth=0.8, linestyle="--", alpha=0.35, zorder=1)
    ax2.axhline(85, color="purple", linewidth=0.5, linestyle=":", alpha=0.3)
    ax2.axhline(15, color="purple", linewidth=0.5, linestyle=":", alpha=0.3)
    ax2.set_ylabel("RV Percentile (%)", fontsize=9, color="purple", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="purple", labelsize=8)
    for label in ax2.get_yticklabels():
        label.set_alpha(0.4)

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    span_days = (common[-1] - common[0]).days if len(common) > 1 else 0
    if span_days <= 120:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif is_zoomed:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ============================================================
    # Panel 2: 买入统计 (Buy Call + Sell Put 合并)
    # ============================================================
    stats = stats_all

    # Panel 2 Left: 短期胜率
    ax = fig.add_subplot(gs[1, 0])
    buy_horizons = [1, 2, 3, 5, 10]
    wr_vals = [stats.get(f"buy_wr_{h}", 0) * 100 for h in buy_horizons]
    colors_buy = ["#85c1e9", "#5dade2", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar([f"{h}d" for h in buy_horizons], wr_vals, color=colors_buy,
                  edgecolor="gray", alpha=0.8, width=0.6)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    for bar, val in zip(bars, wr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"Buy Zone Win Rate (n={stats['buy_n']})", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2 Mid: 止盈命中率
    ax = fig.add_subplot(gs[1, 1])
    tp_windows = [1, 2, 3, 5, 10]
    tp1_vals = [stats.get(f"buy_tp1_{w}d", 0) * 100 for w in tp_windows]
    tp2_vals = [stats.get(f"buy_tp2_{w}d", 0) * 100 for w in tp_windows]
    tp3_vals = [stats.get(f"buy_tp3_{w}d", 0) * 100 for w in tp_windows]
    x = np.arange(len(tp_windows))
    w_bar = 0.25
    bars1 = ax.bar(x - w_bar, tp1_vals, w_bar, label="TP +1%",
                   color="#3498db", edgecolor="gray", alpha=0.8)
    bars2 = ax.bar(x, tp2_vals, w_bar, label="TP +2%",
                   color="#2ecc71", edgecolor="gray", alpha=0.8)
    bars3 = ax.bar(x + w_bar, tp3_vals, w_bar, label="TP +3%",
                   color="#f39c12", edgecolor="gray", alpha=0.8)
    for bars_grp, vals in [(bars1, tp1_vals), (bars2, tp2_vals), (bars3, tp3_vals)]:
        for bar, val in zip(bars_grp, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                        f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}d" for w in tp_windows])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Buy Take-Profit Hit Rate (High)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2 Right: 收益分布
    ax = fig.add_subplot(gs[1, 2])
    fwd_5d = stats.get("buy_fwd_5d", pd.Series(dtype=float))
    fwd_10d = stats.get("buy_fwd_10d", pd.Series(dtype=float))
    if len(fwd_5d) > 0 and len(fwd_10d) > 0:
        ax.hist(fwd_5d, bins=20, alpha=0.5, color="#3498db",
                label=f"5d (n={len(fwd_5d)})", edgecolor="gray")
        ax.hist(fwd_10d, bins=20, alpha=0.5, color="#2ecc71",
                label=f"10d (n={len(fwd_10d)})", edgecolor="gray")
        ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
        ax.axvline(fwd_5d.mean(), color="#3498db", linewidth=2,
                   label=f"5d avg={fwd_5d.mean():+.2f}%")
        ax.axvline(fwd_10d.mean(), color="#2ecc71", linewidth=2,
                   label=f"10d avg={fwd_10d.mean():+.2f}%")
    ax.set_xlabel("Forward Return (%)")
    ax.set_ylabel("Count")
    ax.set_title("Buy Zone Return Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # Panel 3: 卖出统计
    # ============================================================
    # Panel 3 Left: 未大涨率
    ax = fig.add_subplot(gs[2, 0])
    sell_horizons = [1, 2, 3, 5, 10]
    nr1_vals = [stats.get(f"sell_nr1_{h}", 0) * 100 for h in sell_horizons]
    colors_sell = ["#f1948a", "#e74c3c", "#c0392b", "#922b21", "#641e16"]
    bars = ax.bar([f"{h}d" for h in sell_horizons], nr1_vals, color=colors_sell,
                  edgecolor="gray", alpha=0.8, width=0.6)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    for bar, val in zip(bars, nr1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Not-Rally Rate (%)")
    ax.set_title(f"Sell: P(fwd < +1%) (n={stats['sell_n']})", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3 Mid: 平均收益
    ax = fig.add_subplot(gs[2, 1])
    sell_avg_vals = [stats.get(f"sell_avg_{h}", 0) for h in sell_horizons]
    bars = ax.bar([f"{h}d" for h in sell_horizons], sell_avg_vals, color=colors_sell,
                  edgecolor="gray", alpha=0.8, width=0.6)
    ax.axhline(0, color="gray", linewidth=1)
    for bar, val in zip(bars, sell_avg_vals):
        y_pos = val - 0.1 if val < 0 else val + 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Post-Signal Return (%)")
    ax.set_title("Sell Avg Fwd Return (lower=better)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3 Right: 分布
    ax = fig.add_subplot(gs[2, 2])
    sell_fwd_5d = stats.get("sell_fwd_5d", pd.Series(dtype=float))
    sell_fwd_10d = stats.get("sell_fwd_10d", pd.Series(dtype=float))
    if len(sell_fwd_5d) > 0 and len(sell_fwd_10d) > 0:
        ax.hist(sell_fwd_5d, bins=20, alpha=0.5, color="#e74c3c",
                label=f"5d (n={len(sell_fwd_5d)})", edgecolor="gray")
        ax.hist(sell_fwd_10d, bins=20, alpha=0.5, color="#641e16",
                label=f"10d (n={len(sell_fwd_10d)})", edgecolor="gray")
        ax.axvline(0, color="blue", linewidth=1.5, linestyle="--")
        ax.axvline(1.0, color="orange", linewidth=1.5, linestyle="--",
                   label="+1% threshold")
        ax.axvline(sell_fwd_5d.mean(), color="#e74c3c", linewidth=2,
                   label=f"5d avg={sell_fwd_5d.mean():+.2f}%")
    ax.set_xlabel("Forward Return (%)")
    ax.set_ylabel("Count")
    ax.set_title("Sell Signal Return Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 保存
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def _plot_price_panel(ax, gld_close, ub, lb, regime, rv_pctile,
                      buy_call, sell_put, sell, title,
                      date_start, date_end, show_tp=False, high=None,
                      trades=None):
    """复用的价格面板: 价格 + band + 信号 + RV + 可选TP标记 + trades."""
    close = gld_close
    reg = regime.reindex(close.index)
    bp_local = (close - lb) / (ub - lb)
    common = close.index[close.index.isin(bp_local.dropna().index)]
    if date_start:
        common = common[common >= pd.Timestamp(date_start)]
    if date_end:
        common = common[common <= pd.Timestamp(date_end)]
    cl = close.reindex(common)
    ub_c = ub.reindex(common)
    lb_c = lb.reindex(common)

    ax.plot(common, cl, color="black", linewidth=1.2, label="GLD Close", zorder=3)
    ax.fill_between(common, lb_c, ub_c, alpha=0.08, color="steelblue",
                    label="DL Range Band")

    # Regime shading
    reg_c = reg.reindex(common)
    bull = reg_c == "Bull"
    starts = common[bull & (~bull.shift(1, fill_value=False))]
    ends = common[bull & (~bull.shift(-1, fill_value=False))]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.06, color="green")

    rv_s = rv_pctile.reindex(common)
    mk_size = 160

    # Buy Call
    bc_idx = buy_call[buy_call].index.intersection(common)
    if len(bc_idx) > 0:
        ax.scatter(bc_idx, cl.reindex(bc_idx), marker="^", s=mk_size,
                   color="#2196F3", edgecolors="darkblue", linewidths=0.8,
                   zorder=5, label=f"Buy Call ({len(bc_idx)})")
        for d in bc_idx:
            rv_val = rv_s.get(d, np.nan)
            rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
            ax.annotate(f"{d.strftime('%m/%d')}{rv_txt}",
                        xy=(d, cl.loc[d]), xytext=(0, -18),
                        textcoords="offset points", fontsize=6.5,
                        ha="center", color="#2196F3", fontweight="bold")

    # Sell Put
    sp_idx = sell_put[sell_put].index.intersection(common)
    if len(sp_idx) > 0:
        ax.scatter(sp_idx, cl.reindex(sp_idx), marker="^", s=mk_size,
                   color="#FF9800", edgecolors="#E65100", linewidths=0.8,
                   zorder=5, label=f"Sell Put ({len(sp_idx)})")
        for d in sp_idx:
            rv_val = rv_s.get(d, np.nan)
            rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
            ax.annotate(f"{d.strftime('%m/%d')}{rv_txt}",
                        xy=(d, cl.loc[d]), xytext=(0, -18),
                        textcoords="offset points", fontsize=6.5,
                        ha="center", color="#FF9800", fontweight="bold")

    # Exit
    sl_idx = sell[sell].index.intersection(common)
    if len(sl_idx) > 0:
        ax.scatter(sl_idx, cl.reindex(sl_idx), marker="v", s=mk_size,
                   color="#F44336", edgecolors="darkred", linewidths=0.8,
                   zorder=5, label=f"Exit ({len(sl_idx)})")
        for d in sl_idx:
            ax.annotate(f"{d.strftime('%m/%d')}",
                        xy=(d, cl.loc[d]), xytext=(0, 12),
                        textcoords="offset points", fontsize=6.5,
                        ha="center", color="#F44336", fontweight="bold")

    # Trade exit markers: entry→exit with colored arrows
    if trades is not None:
        tp_colors = {"MACD": "#9C27B0", "MACDweak": "#E040FB",
                     "Pullback": "#FF6F00", "RSI": "#00BCD4",
                     "Timeout": "#757575"}
        tp_markers = {"MACD": "D", "MACDweak": "v",
                      "Pullback": "s", "RSI": "h", "Timeout": "X"}
        labeled = set()
        for t in trades:
            ex_d = t["exit"]
            if ex_d not in common:
                continue
            etype = t["exit_type"]
            c = tp_colors.get(etype, "gray")
            m = tp_markers.get(etype, "x")
            lbl = f"TP:{etype}" if etype not in labeled else None
            labeled.add(etype)
            ex_price = cl.get(ex_d, t["exit_price"])
            ax.scatter([ex_d], [ex_price], marker=m, s=120,
                       color=c, edgecolors="black", linewidths=0.5,
                       zorder=7, label=lbl)
            # 细线连接 entry → exit
            en_d = t["entry"]
            if en_d in common:
                en_price = cl.get(en_d, t["entry_price"])
                gain = t["gain_pct"]
                ax.plot([en_d, ex_d], [en_price, ex_price],
                        color=c, linewidth=1.2, alpha=0.5, zorder=2)
                ax.annotate(f"{gain:+.1f}%",
                            xy=(ex_d, ex_price), xytext=(6, 8),
                            textcoords="offset points", fontsize=6.5,
                            color=c, fontweight="bold")

    # RV twin axis
    ax2 = ax.twinx()
    rv_plot = rv_s.dropna()
    ax2.plot(rv_plot.index, rv_plot.values * 100, color="purple",
             linewidth=0.8, linestyle="--", alpha=0.35, zorder=1)
    ax2.axhline(85, color="purple", linewidth=0.5, linestyle=":", alpha=0.3)
    ax2.axhline(15, color="purple", linewidth=0.5, linestyle=":", alpha=0.3)
    ax2.set_ylabel("RV%", fontsize=8, color="purple", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="purple", labelsize=7)
    for lab in ax2.get_yticklabels():
        lab.set_alpha(0.4)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("GLD ($)", fontsize=9)

    span_days = (common[-1] - common[0]).days if len(common) > 1 else 0
    if span_days <= 120:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    return common


def plot_comparison(gld, regime, rv_pctile, close, high, dates,
                    ub_old, lb_old, bp_old, bc_old, sp_old, sl_old,
                    ub_new, lb_new, bp_new, bc_new, sp_new, sl_new,
                    date_start, date_end, filename, trades_new=None):
    """上下对比: old LagAvg vs new Hybrid, 同一时段."""
    fig, axes = plt.subplots(2, 1, figsize=(22, 14), sharex=False)

    d_mask = (dates >= pd.Timestamp(date_start)) & (dates <= pd.Timestamp(date_end))

    _plot_price_panel(
        axes[0], close, ub_old, lb_old, regime, rv_pctile,
        bc_old[d_mask], sp_old[d_mask], sl_old[d_mask],
        f"LagAvg (upper=Lag3, lower=Lag3) — {date_start[:7]} ~ {date_end[:7]}",
        date_start, date_end)

    _plot_price_panel(
        axes[1], close, ub_new, lb_new, regime, rv_pctile,
        bc_new[d_mask], sp_new[d_mask], sl_new[d_mask],
        f"Hybrid (upper=Daily, lower=LagAvg) — {date_start[:7]} ~ {date_end[:7]}",
        date_start, date_end, trades=trades_new)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_tp_analysis(close, high, trades, filename, date_start=None, date_end=None):
    """止盈拐点分析专用图: 上=价格+交易线, 下=统计."""
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        print("  No trades to plot")
        return

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Panel 1: 价格 + 交易线
    ax = fig.add_subplot(gs[0, :])
    common = close.index
    if date_start:
        common = common[common >= pd.Timestamp(date_start)]
    if date_end:
        common = common[common <= pd.Timestamp(date_end)]
    cl = close.reindex(common)
    ax.plot(common, cl, color="black", linewidth=1.2, label="GLD Close", zorder=2)

    tp_colors = {"MACD": "#9C27B0", "MACDweak": "#E040FB",
                 "Pullback": "#FF6F00", "RSI": "#00BCD4",
                 "Timeout": "#757575"}
    tp_markers = {"MACD": "D", "MACDweak": "v",
                  "Pullback": "s", "RSI": "h", "Timeout": "X"}
    labeled = set()

    for _, t in trade_df.iterrows():
        en_d, ex_d = t["entry"], t["exit"]
        if date_start and en_d < pd.Timestamp(date_start):
            continue
        if date_end and en_d > pd.Timestamp(date_end):
            continue
        etype = t["exit_type"]
        c = tp_colors.get(etype, "gray")
        m = tp_markers.get(etype, "x")

        # entry marker
        ax.scatter([en_d], [t["entry_price"]], marker="^", s=100,
                   color="#2196F3", edgecolors="darkblue", linewidths=0.5, zorder=5)
        # exit marker
        lbl = f"TP:{etype}" if etype not in labeled else None
        labeled.add(etype)
        ax.scatter([ex_d], [t["exit_price"]], marker=m, s=120,
                   color=c, edgecolors="black", linewidths=0.5, zorder=6, label=lbl)
        # 连线
        ax.plot([en_d, ex_d], [t["entry_price"], t["exit_price"]],
                color=c, linewidth=1.5, alpha=0.6, zorder=3)
        # 标注收益
        ax.annotate(f"{t['gain_pct']:+.1f}% ({int(t['hold_days'])}d)",
                    xy=(ex_d, t["exit_price"]), xytext=(4, 10),
                    textcoords="offset points", fontsize=6.5,
                    color=c, fontweight="bold")

    ax.set_title(f"TP Exit Analysis: MACD / Pullback / RSI (n={len(trade_df)})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("GLD ($)")
    ax.grid(True, alpha=0.3)

    span_days = (common[-1] - common[0]).days if len(common) > 1 else 0
    if span_days <= 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 2 Left: 退出类型分布
    ax = fig.add_subplot(gs[1, 0])
    type_counts = trade_df["exit_type"].value_counts()
    colors_bar = [tp_colors.get(t, "gray") for t in type_counts.index]
    bars = ax.bar(type_counts.index, type_counts.values, color=colors_bar,
                  edgecolor="gray", alpha=0.8)
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                str(val), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Exit Type Distribution", fontsize=11)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2 Mid: 各类型收益
    ax = fig.add_subplot(gs[1, 1])
    for etype in ["MACD", "MACDweak", "Pullback", "RSI", "Timeout"]:
        sub = trade_df[trade_df["exit_type"] == etype]
        if len(sub) == 0:
            continue
        ax.bar(etype, sub["gain_pct"].mean(),
               color=tp_colors[etype], edgecolor="gray", alpha=0.8)
        ax.text(ax.patches[-1].get_x() + ax.patches[-1].get_width() / 2,
                sub["gain_pct"].mean() + 0.1,
                f"{sub['gain_pct'].mean():+.2f}%\n(n={len(sub)})",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Avg Gain by Exit Type", fontsize=11)
    ax.set_ylabel("Gain (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2 Right: 持仓天数 vs 收益散点
    ax = fig.add_subplot(gs[1, 2])
    for etype in ["MACD", "MACDweak", "Pullback", "RSI", "Timeout"]:
        sub = trade_df[trade_df["exit_type"] == etype]
        if len(sub) == 0:
            continue
        ax.scatter(sub["hold_days"], sub["gain_pct"],
                   color=tp_colors[etype], edgecolors="black",
                   s=60, alpha=0.7, label=etype)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Hold Days")
    ax.set_ylabel("Gain (%)")
    ax.set_title("Hold Days vs Gain", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==========================================================
# main
# ==========================================================
def main():
    print("=" * 70)
    print("  方法对比: 基线 A vs 优化 V2 (期权视角)")
    print("=" * 70)

    gld, range_df, regime, rv_10d = load_all()
    # Hybrid D/L3: upper=Daily(t-1), lower=LagAvg(t-1,t-2,t-3)
    upper_band, lower_band, bp = build_band(
        range_df, gld["Close"], upper_lags=(1,), lower_lags=(1, 2, 3))
    rv_pctile = compute_rv_pctile(rv_10d)
    close = gld["Close"]
    high = gld["High"]

    dates = bp.dropna().index
    reg = regime.reindex(dates)
    bp_s = bp.reindex(dates)
    rv_p = rv_pctile.reindex(dates)
    is_bull = reg == "Bull"
    span = (dates[-1] - dates[0]).days / 365.25

    # === V2 信号 (水平触发 + 类型区分) ===
    buy_call, sell_put, sell_V2 = generate_v2_signals(bp_s, rv_p, is_bull)

    buy_all = buy_call | sell_put  # 买入区 (统计用)
    print(f"\n  V2 (水平触发, 期权类型):")
    print(f"    Buy Call: {buy_call.sum()}次 ({buy_call.sum()/span:.1f}/年)")
    print(f"    Sell Put: {sell_put.sum()}次 ({sell_put.sum()/span:.1f}/年)")
    print(f"    Sell/TP:  {sell_V2.sum()}次 ({sell_V2.sum()/span:.1f}/年)")

    # 分开统计 buy_call 和 sell_put
    stats_bc = eval_signals(buy_call, sell_V2, close, high=high)
    stats_sp = eval_signals(sell_put, sell_V2, close, high=high)
    stats_all = eval_signals(buy_all, sell_V2, close, high=high)

    print(f"\n  Buy Call (Bull+bp<0.30+RV≤85%) n={stats_bc['buy_n']}:")
    print(f"    5d WR={stats_bc.get('buy_wr_5',0):.0%}  "
          f"5d TP+1%={stats_bc.get('buy_tp1_5d',0):.0%}  "
          f"5d TP+2%={stats_bc.get('buy_tp2_5d',0):.0%}")
    print(f"  Sell Put (Bull+bp<0.30+RV>85%) n={stats_sp['buy_n']}:")
    print(f"    5d WR={stats_sp.get('buy_wr_5',0):.0%}  "
          f"5d TP+1%={stats_sp.get('buy_tp1_5d',0):.0%}  "
          f"5d TP+2%={stats_sp.get('buy_tp2_5d',0):.0%}")
    print(f"  Sell/TP n={stats_all['sell_n']}:")
    print(f"    1d NR={stats_all.get('sell_nr1_1',0):.0%}  "
          f"3d NR={stats_all.get('sell_nr1_3',0):.0%}  "
          f"5d NR={stats_all.get('sell_nr1_5',0):.0%}")

    # === 方法 A: 基线 (保留穿越触发作为对照) ===
    buy_A, sell_A = generate_baseline_signals(bp_s, is_bull)
    stats_A = eval_signals(buy_A, sell_A, close, high=high)
    print(f"\n  方法 A (基线, 穿越触发): buy={stats_A['buy_n']}, sell={stats_A['sell_n']}")

    # === 生成可视化 ===
    print(f"\n  生成可视化...")

    # 全时段 V2 Hybrid
    plot_v2(gld, regime, upper_band, lower_band, bp, rv_pctile,
            buy_call, sell_put, sell_V2, stats_all,
            "V2 Hybrid (upper=Daily, lower=LagAvg)",
            "15_method_V2_optimized.png")

    # 近一年
    oos_end = dates[-1]
    zoom_end_dt = oos_end + pd.Timedelta(days=7)
    zoom_start_dt = oos_end - pd.Timedelta(days=365)
    zoom_start = zoom_start_dt.strftime("%Y-%m-%d")
    zoom_end = zoom_end_dt.strftime("%Y-%m-%d")
    print(f"\n  近一年 ({zoom_start} ~ {zoom_end})...")

    z_mask = (dates >= pd.Timestamp(zoom_start)) & (dates <= pd.Timestamp(zoom_end))
    stats_z = eval_signals(buy_all[z_mask], sell_V2[z_mask], close, high=high)
    stats_bc_z = eval_signals(buy_call[z_mask], sell_V2[z_mask], close, high=high)
    stats_sp_z = eval_signals(sell_put[z_mask], sell_V2[z_mask], close, high=high)

    print(f"    Buy Call={buy_call[z_mask].sum()}, Sell Put={sell_put[z_mask].sum()}, "
          f"Sell={sell_V2[z_mask].sum()}")

    plot_v2(gld, regime, upper_band, lower_band, bp, rv_pctile,
            buy_call[z_mask], sell_put[z_mask], sell_V2[z_mask], stats_z,
            f"V2 — {zoom_start_dt.strftime('%Y.%m')} ~ {oos_end.strftime('%Y.%m')}",
            "17_method_V2_recent.png",
            date_start=zoom_start, date_end=zoom_end)

    # 2026 Q1
    q1_start, q1_end = "2026-01-01", "2026-03-31"
    q1_mask = (dates >= pd.Timestamp(q1_start)) & (dates <= pd.Timestamp(q1_end))
    stats_q1 = eval_signals(buy_all[q1_mask], sell_V2[q1_mask], close, high=high)

    print(f"\n  2026 Q1: Buy Call={buy_call[q1_mask].sum()}, "
          f"Sell Put={sell_put[q1_mask].sum()}, Sell={sell_V2[q1_mask].sum()}")

    plot_v2(gld, regime, upper_band, lower_band, bp, rv_pctile,
            buy_call[q1_mask], sell_put[q1_mask], sell_V2[q1_mask], stats_q1,
            "V2 Hybrid — 2026 Q1 (Jan~Feb)",
            "19_method_V2_2026Q1.png",
            date_start=q1_start, date_end=q1_end)

    # ============================================================
    # 2025-10 ~ 2026-02 Hybrid vs LagAvg 对比
    # ============================================================
    cmp_start, cmp_end = "2025-10-01", "2026-03-01"
    cmp_mask = (dates >= pd.Timestamp(cmp_start)) & (dates <= pd.Timestamp(cmp_end))

    # 旧 LagAvg band
    ub_old, lb_old, bp_old = build_band(range_df, gld["Close"], lags=(1, 2, 3))
    bp_old_s = bp_old.reindex(dates)
    is_bull_s = is_bull.copy()
    bc_old, sp_old, sl_old = generate_v2_signals(bp_old_s, rv_p, is_bull_s)
    buy_old = bc_old | sp_old
    stats_old_cmp = eval_signals(buy_old[cmp_mask], sl_old[cmp_mask], close, high=high)
    stats_new_cmp = eval_signals(buy_all[cmp_mask], sell_V2[cmp_mask], close, high=high)

    print(f"\n  === 2025-10 ~ 2026-02: Hybrid vs LagAvg ===")
    print(f"    {'':20s} {'LagAvg':>10s} {'Hybrid':>10s}")
    print(f"    {'-'*42}")
    for label, key in [
        ("Buy signals", "buy_n"), ("Sell signals", "sell_n"),
    ]:
        print(f"    {label:20s} {stats_old_cmp.get(key,0):>10d} {stats_new_cmp.get(key,0):>10d}")
    for h in [1, 3, 5]:
        wo = stats_old_cmp.get(f'buy_wr_{h}', 0) * 100
        wn = stats_new_cmp.get(f'buy_wr_{h}', 0) * 100
        print(f"    {'Buy %dd WR' % h:20s} {wo:>9.0f}% {wn:>9.0f}%")
    for h in [1, 3, 5]:
        no = stats_old_cmp.get(f'sell_nr1_{h}', 0) * 100
        nn = stats_new_cmp.get(f'sell_nr1_{h}', 0) * 100
        print(f"    {'Exit %dd NR' % h:20s} {no:>9.0f}% {nn:>9.0f}%")

    # ============================================================
    # 止盈拐点分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  止盈拐点分析 (MACD/Pullback/RSI)")
    print(f"{'='*70}")

    low = gld["Low"]
    tp_ind = compute_tp_indicators(close, high, low)

    # 全量买入信号的止盈退出
    buy_dates_all = buy_all[buy_all].index
    trades_all = find_tp_exits(buy_dates_all, close, high, tp_ind, max_hold=10)
    trade_df = pd.DataFrame(trades_all)

    if len(trade_df) > 0:
        print(f"\n  全量交易: {len(trade_df)} trades")
        print(f"  {'Type':10s} {'N':>5s} {'AvgGain':>8s} {'WinRate':>8s} "
              f"{'AvgHold':>8s} {'AvgPeak':>8s} {'Capture':>8s}")
        print(f"  {'-'*58}")
        for etype in ["MACD", "MACDweak", "Pullback", "RSI", "Timeout"]:
            sub = trade_df[trade_df["exit_type"] == etype]
            if len(sub) == 0:
                continue
            wr = (sub["gain_pct"] > 0).mean() * 100
            capture = (sub["gain_pct"] / sub["peak_pct"].clip(lower=0.01)).mean() * 100
            print(f"  {etype:10s} {len(sub):5d} {sub['gain_pct'].mean():+7.2f}% "
                  f"{wr:7.0f}% {sub['hold_days'].mean():7.1f}d "
                  f"{sub['peak_pct'].mean():+7.2f}% {capture:7.0f}%")

        total_wr = (trade_df["gain_pct"] > 0).mean() * 100
        total_capture = (trade_df["gain_pct"] / trade_df["peak_pct"].clip(lower=0.01)).mean() * 100
        print(f"  {'TOTAL':10s} {len(trade_df):5d} {trade_df['gain_pct'].mean():+7.2f}% "
              f"{total_wr:7.0f}% {trade_df['hold_days'].mean():7.1f}d "
              f"{trade_df['peak_pct'].mean():+7.2f}% {total_capture:7.0f}%")

        # 与固定阈值对比
        print(f"\n  vs 固定止盈对比:")
        for tp_level in [1.0, 2.0]:
            # 固定: 5天内达到+N%就算止盈
            tp_hit = 0
            tp_gains = []
            for _, t in trade_df.iterrows():
                entry_loc = close.index.get_loc(t["entry"])
                window = close.index[entry_loc + 1: entry_loc + 6]
                hh = high.reindex(window)
                target = t["entry_price"] * (1 + tp_level / 100)
                if (hh >= target).any():
                    tp_hit += 1
                    tp_gains.append(tp_level)
                else:
                    # 没达到就按5d close算
                    if len(window) > 0:
                        exit_p = close.get(window[-1], t["entry_price"])
                        tp_gains.append((exit_p / t["entry_price"] - 1) * 100)
            fixed_avg = np.mean(tp_gains) if tp_gains else 0
            print(f"    Fixed +{tp_level:.0f}%: hit={tp_hit}/{len(trade_df)} "
                  f"({tp_hit/len(trade_df)*100:.0f}%), avg={fixed_avg:+.2f}%")
        print(f"    Smart TP:  avg={trade_df['gain_pct'].mean():+.2f}%, "
              f"WR={total_wr:.0f}%, capture={total_capture:.0f}%")

    # 对比可视化: 上下两排
    print(f"\n  生成 Hybrid vs LagAvg 对比图 (2025.10~2026.02)...")

    # 区间内的trades
    cmp_trades = [t for t in trades_all
                  if pd.Timestamp(cmp_start) <= t["entry"] <= pd.Timestamp(cmp_end)]

    plot_comparison(
        gld, regime, rv_pctile, close, high, dates,
        # old LagAvg
        ub_old, lb_old, bp_old, bc_old, sp_old, sl_old,
        # new Hybrid
        upper_band, lower_band, bp, buy_call, sell_put, sell_V2,
        date_start=cmp_start, date_end=cmp_end,
        filename="20_hybrid_vs_lagavg.png",
        trades_new=cmp_trades)

    # TP 专题图: 2025-10 ~ 2026-02
    plot_tp_analysis(close, high, cmp_trades,
                     "21_tp_exit_analysis.png",
                     date_start=cmp_start, date_end=cmp_end)

    # TP 专题图: 全量
    plot_tp_analysis(close, high, trades_all,
                     "22_tp_exit_analysis_full.png")

    # === 期权视角详细分析 ===
    print(f"\n{'='*70}")
    print(f"  期权视角详细分析")
    print(f"{'='*70}")

    for label, s in [("Buy Call (RV≤85%)", stats_bc),
                     ("Sell Put (RV>85%)", stats_sp),
                     ("合计买入区", stats_all)]:
        if s["buy_n"] == 0:
            print(f"\n  --- {label}: 无信号 ---")
            continue
        print(f"\n  --- {label} (n={s['buy_n']}) ---")

        print(f"    胜率:  ", end="")
        for h in [1, 3, 5, 10]:
            print(f"{h}d={s.get(f'buy_wr_{h}',0):.0%}  ", end="")
        print()

        print(f"    止盈命中率 (High):")
        print(f"    {'Window':>8s} {'≥+1%':>8s} {'≥+2%':>8s} {'≥+3%':>8s} {'AvgMax':>8s}")
        for w in [1, 2, 3, 5, 10]:
            tp1 = s.get(f'buy_tp1_{w}d', 0)
            tp2 = s.get(f'buy_tp2_{w}d', 0)
            tp3 = s.get(f'buy_tp3_{w}d', 0)
            mg = s.get(f'buy_maxgain_{w}d', 0)
            print(f"    {f'{w}d':>8s} {tp1:>7.0%} {tp2:>7.0%} {tp3:>7.0%} {mg:>+7.2f}%")

    print(f"\n  --- 卖出 (n={stats_all['sell_n']}) ---")
    print(f"    未大涨率 P(fwd < +1%):")
    for h in [1, 2, 3, 5, 10]:
        nr1 = stats_all.get(f'sell_nr1_{h}', 0)
        nr2 = stats_all.get(f'sell_nr2_{h}', 0)
        avg = stats_all.get(f'sell_avg_{h}', 0)
        print(f"      {h}d: <+1%={nr1:.0%}  <+2%={nr2:.0%}  avg={avg:+.2f}%")

    # === V2 vs 基线对比 ===
    print(f"\n{'='*70}")
    print(f"  V2 (水平触发) vs 基线 A (穿越触发)")
    print(f"{'='*70}")
    print(f"  {'':30s} {'基线A':>10s} {'V2合计':>10s} {'BuyCall':>10s} {'SellPut':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'买入信号数':30s} {stats_A['buy_n']:>10d} {stats_all['buy_n']:>10d} "
          f"{stats_bc['buy_n']:>10d} {stats_sp['buy_n']:>10d}")
    for h in [1, 3, 5]:
        wa = stats_A.get(f'buy_wr_{h}', 0) * 100
        wv = stats_all.get(f'buy_wr_{h}', 0) * 100
        wc = stats_bc.get(f'buy_wr_{h}', 0) * 100
        wp = stats_sp.get(f'buy_wr_{h}', 0) * 100
        print(f"  {'买入 %dd 胜率' % h:30s} {wa:>9.0f}% {wv:>9.0f}% {wc:>9.0f}% {wp:>9.0f}%")
    for w in [3, 5]:
        ta = stats_A.get(f'buy_tp2_{w}d', 0) * 100
        tv = stats_all.get(f'buy_tp2_{w}d', 0) * 100
        tc = stats_bc.get(f'buy_tp2_{w}d', 0) * 100
        tp = stats_sp.get(f'buy_tp2_{w}d', 0) * 100
        print(f"  {'买入 %dd 止盈≥+2%%' % w:30s} {ta:>9.0f}% {tv:>9.0f}% {tc:>9.0f}% {tp:>9.0f}%")

    print(f"  {'':30s} {'':>10s} {'':>10s}")
    print(f"  {'卖出信号数':30s} {stats_A['sell_n']:>10d} {stats_all['sell_n']:>10d}")
    for h in [1, 3, 5]:
        na = stats_A.get(f'sell_nr1_{h}', 0) * 100
        nv = stats_all.get(f'sell_nr1_{h}', 0) * 100
        print(f"  {'卖出 %dd 未大涨率' % h:30s} {na:>9.0f}% {nv:>9.0f}%")

    # ==========================================================
    # 区间重算方式对比: Daily vs Lag-Avg(1,2,3) vs Lag(1,2)
    # ==========================================================
    print(f"\n{'='*70}")
    print(f"  区间重算方式对比 (Daily vs Lag-Avg)")
    print(f"{'='*70}")

    band_configs = [
        # (name, lags, upper_lags, lower_lags)
        ("Daily (t-1)",     (1,),        None, None),
        ("Lag2 (t-1,t-2)",  (1, 2),      None, None),
        ("LagAvg (t-1~3)",  (1, 2, 3),   None, None),
        ("Lag4 (t-1~4)",    (1, 2, 3, 4), None, None),
        # Hybrid: upper=Daily, lower=LagAvg
        ("Hybrid D/L3",     (1,),        (1,), (1, 2, 3)),
        # Hybrid: upper=Daily, lower=Lag2
        ("Hybrid D/L2",     (1,),        (1,), (1, 2)),
        # Hybrid: upper=Lag2, lower=LagAvg
        ("Hybrid L2/L3",    (1, 2),      (1, 2), (1, 2, 3)),
    ]

    results_by_band = {}
    for bname, lags, u_lags, l_lags in band_configs:
        ub_b, lb_b, bp_b = build_band(range_df, gld["Close"], lags=lags,
                                       upper_lags=u_lags, lower_lags=l_lags)
        d_b = bp_b.dropna().index
        bp_bs = bp_b.reindex(d_b)
        rv_pb = rv_pctile.reindex(d_b)
        is_bull_b = (regime.reindex(d_b) == "Bull")

        bc_b, sp_b, sl_b = generate_v2_signals(bp_bs, rv_pb, is_bull_b)
        buy_b = bc_b | sp_b
        s_b = eval_signals(buy_b, sl_b, close, high=high)
        s_bc = eval_signals(bc_b, sl_b, close, high=high)
        s_sp = eval_signals(sp_b, sl_b, close, high=high)
        results_by_band[bname] = {
            "all": s_b, "bc": s_bc, "sp": s_sp,
            "n_bc": bc_b.sum(), "n_sp": sp_b.sum(), "n_sell": sl_b.sum(),
        }

    # 打印对比表
    names = [n for n, *_ in band_configs]
    print(f"\n  {'':28s}", end="")
    for n in names:
        print(f" {n:>16s}", end="")
    print()
    print(f"  {'-'*92}")

    # 买入区
    for metric_label, key, is_count in [
        ("Buy区 信号数", lambda r: r["all"]["buy_n"], True),
        ("  Buy Call", lambda r: r["n_bc"], True),
        ("  Sell Put", lambda r: r["n_sp"], True),
        ("Buy 5d 胜率", lambda r: r["all"].get("buy_wr_5", 0) * 100, False),
        ("Buy 5d TP+1%", lambda r: r["all"].get("buy_tp1_5d", 0) * 100, False),
        ("Buy 5d TP+2%", lambda r: r["all"].get("buy_tp2_5d", 0) * 100, False),
        ("Buy 10d TP+2%", lambda r: r["all"].get("buy_tp2_10d", 0) * 100, False),
        ("BuyCall 5d WR", lambda r: r["bc"].get("buy_wr_5", 0) * 100, False),
        ("SellPut 5d WR", lambda r: r["sp"].get("buy_wr_5", 0) * 100, False),
        ("SellPut 5d TP+2%", lambda r: r["sp"].get("buy_tp2_5d", 0) * 100, False),
    ]:
        print(f"  {metric_label:28s}", end="")
        for n in names:
            v = key(results_by_band[n])
            if is_count:
                print(f" {int(v):>16d}", end="")
            else:
                print(f" {v:>15.0f}%", end="")
        print()

    # 卖出区
    print()
    for metric_label, key, is_count in [
        ("平仓 信号数", lambda r: r["n_sell"], True),
        ("平仓 1d NR(<+1%)", lambda r: r["all"].get("sell_nr1_1", 0) * 100, False),
        ("平仓 3d NR(<+1%)", lambda r: r["all"].get("sell_nr1_3", 0) * 100, False),
        ("平仓 5d NR(<+1%)", lambda r: r["all"].get("sell_nr1_5", 0) * 100, False),
    ]:
        print(f"  {metric_label:28s}", end="")
        for n in names:
            v = key(results_by_band[n])
            if is_count:
                print(f" {int(v):>16d}", end="")
            else:
                print(f" {v:>15.0f}%", end="")
        print()


if __name__ == "__main__":
    main()
