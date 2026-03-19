"""
GLD 期权交易仪表板

Streamlit 交互式界面:
  - 今日预测: 5日区间 + 信号 + 期权策略
  - 历史回看: 自定义时间范围可视化

用法:
    conda activate gold
    streamlit run app.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

warnings.filterwarnings("ignore")

from core.data import (load_config, load_features, load_gld,
                       load_oos_predictions, load_latest_eod_snapshot,
                       load_all_eod_snapshots, load_gold_futures,
                       load_usdcny, fetch_realtime_gold_fx,
                       auto_refresh_market_data, get_today_sgt,
                       update_features_incremental, extend_oos_predictions)
from core.regime import RegimeClassifier
from core.signals import build_band, compute_rv_pctile, generate_signals
from core.signals_1h import build_band_1h, generate_signals_1h, backtest_1h
from core.options import get_strategy_table
from core.oi_factors import (compute_oi_factors, adjust_range,
                             adjust_range_daily, adjust_band_history)

# ── 中文字体 ──
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang HK",
                                "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="GLD 交易仪表板", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ── 时区配置 ──
TZ_OPTIONS = {"SGT (UTC+8)": 8, "ET (UTC-5)": -5, "UTC": 0, "CST (UTC+8)": 8}
TZ_DEFAULT = "SGT (UTC+8)"

SIG_COLORS = {"BUY CALL": "#2196F3", "SELL PUT": "#FF9800"}
EXIT_MARKERS = {
    "BandExit": ("v", "#F44336"),
    "Pullback": ("s", "#FF6600"),
    "Timeout":  ("X", "gray"),
}


# ══════════════════════════════════════════════════════════
# 缓存数据加载
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_all():
    """加载全部数据, 缓存1小时."""
    cfg = load_config()
    features = load_features(cfg)
    gld = load_gld(cfg)
    range_df = load_oos_predictions(cfg)

    common = features.index.intersection(gld.index)
    features = features.loc[common]
    gld = gld.loc[common]

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    rv_10d = features["rv_10d"] if "rv_10d" in features.columns else None
    rv_pctile = compute_rv_pctile(rv_10d)

    # 换算数据: GC/GLD 比例 + USD/CNY
    gc_gld_ratio = None
    usdcny_rate = None
    gc_df = load_gold_futures(cfg)
    if gc_df is not None:
        gc_common = gld.index.intersection(gc_df.index)
        if len(gc_common) > 20:
            ratios = gc_df.loc[gc_common[-60:], "Close"] / \
                     gld.loc[gc_common[-60:], "Close"]
            gc_gld_ratio = float(ratios.mean())
    usdcny_s = load_usdcny(cfg)
    if usdcny_s is not None and len(usdcny_s) > 0:
        usdcny_rate = float(usdcny_s.iloc[-1])

    return gld, range_df, regime, rv_pctile, gc_gld_ratio, usdcny_rate


@st.cache_data(ttl=300)
def _get_realtime_prices():
    """实时金价+汇率 (5分钟缓存)."""
    return fetch_realtime_gold_fx()


@st.cache_data(ttl=3600)
def load_1h_data():
    """加载 GC=F 1h 数据和 OOS 预测.

    GC=F 为主信号源 (全球24h), GLD 为跨市场参考.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "Gold", "data", "raw", "market")
    data_dir = os.path.normpath(data_dir)
    model_dir = os.path.join(os.path.dirname(data_dir), "..", "models")
    model_dir = os.path.normpath(model_dir)

    gc_path = os.path.join(data_dir, "gc_1h.csv")
    gld_path = os.path.join(data_dir, "gld_1h.csv")
    pred_path = os.path.join(model_dir, "dl_range_1h_oos.parquet")

    if not os.path.exists(gc_path) or not os.path.exists(pred_path):
        return None, None, None
    gc_1h = pd.read_csv(gc_path, index_col=0, parse_dates=True)
    gld_1h = pd.read_csv(gld_path, index_col=0, parse_dates=True) \
        if os.path.exists(gld_path) else None
    pred_1h = pd.read_parquet(pred_path)
    return gc_1h, gld_1h, pred_1h


# ══════════════════════════════════════════════════════════
# 交易构建
# ══════════════════════════════════════════════════════════
def build_trades(close, high, dates_all, buy_call, sell_put, exit_sig,
                 max_hold=10):
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


# ══════════════════════════════════════════════════════════
# 图表
# ══════════════════════════════════════════════════════════
def generate_chart(close, high, dates_all, upper_band, lower_band,
                   buy_call, sell_put, exit_sig, rv_pctile, regime,
                   pred_u_pct=None, pred_l_pct=None,
                   show_future=True, today=None, today_close=None,
                   next_bp030=0, next_bp090=0, signal_type=None,
                   today_rv=0, oi_levels=None, oi_daily_range=None,
                   oi_events=None, oi_adj_bands=None,
                   oi_adj_bp030=0, oi_adj_bp090=0):
    fig, ax = plt.subplots(figsize=(18, 9))
    trades = build_trades(close, high, dates_all, buy_call, sell_put, exit_sig)

    # ── 建立 date→index 映射 (消除周末断层) ──
    plot_dates = list(dates_all)
    future_bdays = []
    if show_future and today is not None and pred_u_pct is not None:
        n_fut = len(oi_daily_range) if oi_daily_range else 5
        future_bdays = list(pd.bdate_range(
            today + timedelta(days=1), periods=n_fut))
        plot_dates = plot_dates + [d for d in future_bdays
                                   if d not in set(plot_dates)]
    d2i = {d: i for i, d in enumerate(plot_dates)}

    def xi(d):
        return d2i.get(d)

    def xi_arr(dates):
        return [d2i[d] for d in dates if d in d2i]

    def _fmt_tick(x, pos):
        idx = int(round(x))
        if 0 <= idx < len(plot_dates):
            return plot_dates[idx].strftime("%m/%d")
        return ""

    # 价格
    cl = close.reindex(dates_all).dropna()
    ax.plot(xi_arr(cl.index), cl.values, "k-", lw=1.8, alpha=0.9, zorder=3)

    # Band (原始)
    ub = upper_band.reindex(dates_all).dropna()
    lb = lower_band.reindex(dates_all).dropna()
    ax.plot(xi_arr(ub.index), ub.values, color="green", lw=1, alpha=0.5)
    ax.plot(xi_arr(lb.index), lb.values, color="magenta", lw=1, alpha=0.5)
    cidx = ub.index.intersection(lb.index)
    if len(cidx) > 0:
        ax.fill_between(xi_arr(cidx),
                         lb.loc[cidx].values, ub.loc[cidx].values,
                         alpha=0.06, color="green")

    # Band (OI 修正)
    if oi_adj_bands is not None:
        adj_ub, adj_lb = oi_adj_bands
        adj_ub = adj_ub.reindex(dates_all).dropna()
        adj_lb = adj_lb.reindex(dates_all).dropna()
        if len(adj_ub) > 0:
            ax.plot(xi_arr(adj_ub.index), adj_ub.values,
                    color="darkgreen", lw=1.5, ls="--", alpha=0.7)
            ax.plot(xi_arr(adj_lb.index), adj_lb.values,
                    color="darkmagenta", lw=1.5, ls="--", alpha=0.7)
            aidx = adj_ub.index.intersection(adj_lb.index)
            if len(aidx) > 0:
                ax.fill_between(xi_arr(aidx),
                                 adj_lb.loc[aidx].values,
                                 adj_ub.loc[aidx].values,
                                 alpha=0.08, color="orange")

    # Regime 背景
    reg = regime.reindex(dates_all)
    bull = reg == "Bull"
    if bull.any():
        starts = dates_all[bull & (~bull.shift(1, fill_value=False))]
        ends = dates_all[bull & (~bull.shift(-1, fill_value=False))]
        for s, e in zip(starts, ends):
            si, ei = xi(s), xi(e)
            if si is not None and ei is not None:
                ax.axvspan(si, ei, alpha=0.04, color="green")

    # Exit 信号
    entry_dates_set = set(t["entry_date"] for t in trades)
    ex_dates = [d for d in dates_all if exit_sig.get(d, False)
                and d not in entry_dates_set]
    if ex_dates:
        ax.scatter(xi_arr(ex_dates),
                   [cl.get(d, np.nan) for d in ex_dates],
                   marker="v", s=120, color="#F44336", edgecolors="darkred",
                   linewidths=0.7, zorder=5)
        for d in ex_dates:
            v = cl.get(d, np.nan)
            if not np.isnan(v) and xi(d) is not None:
                ax.annotate(d.strftime("%m/%d"), xy=(xi(d), v),
                            xytext=(0, 12), textcoords="offset points",
                            fontsize=6, ha="center", color="#F44336",
                            fontweight="bold")

    # 交易轨迹
    for t in trades:
        td = [xi(x[0]) for x in t["trajectory"] if xi(x[0]) is not None]
        tp = [x[1] for x in t["trajectory"] if xi(x[0]) is not None]
        c = SIG_COLORS[t["sig_type"]]
        ax.plot(td, tp, "-", color=c, lw=2,
                alpha=0.85 if t["gain"] > 0 else 0.4, zorder=4)
        ei = xi(t["entry_date"])
        if ei is not None:
            ax.scatter([ei], [t["entry_price"]], marker="^",
                       s=160, color=c, edgecolors="black", linewidths=0.7,
                       zorder=6)
            rv_val = rv_pctile.get(t["entry_date"], np.nan)
            rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
            ax.annotate(
                f"{t['entry_date'].strftime('%m/%d')}{rv_txt}",
                xy=(ei, t["entry_price"]),
                xytext=(0, -16), textcoords="offset points",
                fontsize=7, ha="center", color=c, fontweight="bold")
        exi = xi(t["exit_date"])
        if exi is not None:
            mk, mc = EXIT_MARKERS.get(t["exit_type"], ("o", "gray"))
            if t["exit_date"] not in entry_dates_set:
                ax.scatter([exi], [t["exit_price"]], marker=mk,
                           s=100, color=mc, edgecolors="black",
                           linewidths=0.5, zorder=7)
            oy = 16 if t["exit_date"] in entry_dates_set \
                else (12 if t["gain"] > 0 else -14)
            ax.annotate(f"{t['gain']:+.1f}% ({t['hold_days']}d)",
                        xy=(exi, t["exit_price"]),
                        xytext=(5, oy), textcoords="offset points",
                        fontsize=7, color=c, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.8, ec="none"))

    # RV 副轴
    ax2 = ax.twinx()
    rv_plot = rv_pctile.reindex(dates_all).dropna()
    ax2.plot(xi_arr(rv_plot.index), rv_plot.values * 100, color="purple",
             lw=0.7, ls="--", alpha=0.3, zorder=1)
    ax2.axhline(85, color="purple", lw=0.5, ls=":", alpha=0.3)
    ax2.set_ylabel("RV%", fontsize=7, color="purple", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="purple", labelsize=6)
    for lab in ax2.get_yticklabels():
        lab.set_alpha(0.4)

    # 5日预测区间
    if show_future and today is not None and pred_u_pct is not None:
        tu = today_close * (1 + pred_u_pct / 100)
        tl = today_close * (1 + pred_l_pct / 100)
        fut_xi = xi_arr(future_bdays)

        if oi_daily_range is not None and len(oi_daily_range) > 0 \
                and len(fut_xi) == len(oi_daily_range):
            uppers = [d[0] for d in oi_daily_range]
            lowers = [d[1] for d in oi_daily_range]

            ax.fill_between(fut_xi, lowers, uppers,
                             alpha=0.12, color="gold", zorder=1)
            ax.plot(fut_xi, uppers, color="goldenrod", lw=1.2,
                    ls="--", alpha=0.7)
            ax.plot(fut_xi, lowers, color="goldenrod", lw=1.2,
                    ls="--", alpha=0.7)

            # 原始模型区间 (灰色虚线参考)
            ax.plot([fut_xi[0], fut_xi[-1]], [tu, tu],
                    color="gray", lw=0.8, ls=":", alpha=0.4)
            ax.plot([fut_xi[0], fut_xi[-1]], [tl, tl],
                    color="gray", lw=0.8, ls=":", alpha=0.4)

            # 标注最终日
            ax.annotate(
                f"${uppers[-1]:.0f} ({(uppers[-1]/today_close-1)*100:+.1f}%)",
                xy=(fut_xi[-1], uppers[-1]), fontsize=8, fontweight="bold",
                color="goldenrod", ha="right", va="bottom")
            ax.annotate(
                f"${lowers[-1]:.0f} ({(lowers[-1]/today_close-1)*100:+.1f}%)",
                xy=(fut_xi[-1], lowers[-1]), fontsize=8, fontweight="bold",
                color="goldenrod", ha="right", va="top")

            # OPEX 到期日标记
            if oi_events:
                for day_idx, desc in oi_events:
                    if day_idx <= len(fut_xi):
                        ev_xi = fut_xi[day_idx - 1]
                        ax.axvline(ev_xi, color="orange", lw=1.5,
                                   ls="--", alpha=0.6, zorder=2)
                        ev_y = max(uppers) + \
                            (max(uppers) - min(lowers)) * 0.02
                        ax.annotate(desc, xy=(ev_xi, ev_y),
                                    fontsize=7, color="orange",
                                    fontweight="bold", ha="center",
                                    va="bottom")
        else:
            # 无 OI: 平矩形
            if len(fut_xi) >= 2:
                ax.fill_between([fut_xi[0], fut_xi[-1]],
                                 [tl, tl], [tu, tu],
                                 alpha=0.12, color="gold", zorder=1)
                ax.plot([fut_xi[0], fut_xi[-1]], [tu, tu],
                        color="goldenrod", lw=1.2, ls="--", alpha=0.7)
                ax.plot([fut_xi[0], fut_xi[-1]], [tl, tl],
                        color="goldenrod", lw=1.2, ls="--", alpha=0.7)
                ax.annotate(f"${tu:.0f} (+{pred_u_pct:.1f}%)",
                            xy=(fut_xi[-1], tu), fontsize=8,
                            fontweight="bold", color="goldenrod",
                            ha="right", va="bottom")
                ax.annotate(f"${tl:.0f} ({pred_l_pct:.1f}%)",
                            xy=(fut_xi[-1], tl), fontsize=8,
                            fontweight="bold", color="goldenrod",
                            ha="right", va="top")

    # 下一交易日阈值线
    last_xi = len(dates_all) - 1
    # 使用 OI 修正后的阈值 (如有), 否则用原始值
    eff_bp030 = oi_adj_bp030 if oi_adj_bp030 > 0 else next_bp030
    eff_bp090 = oi_adj_bp090 if oi_adj_bp090 > 0 else next_bp090
    if eff_bp030 > 0:
        ax.axhline(eff_bp030, color="#2196F3", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        buy_label = f"BUY < ${eff_bp030:.1f}"
        if oi_adj_bp030 > 0 and next_bp030 > 0:
            buy_label += f" (原${next_bp030:.1f})"
        ax.annotate(buy_label,
                    xy=(last_xi, eff_bp030),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#2196F3", fontweight="bold",
                    ha="left", va="center")
    if eff_bp090 > 0:
        ax.axhline(eff_bp090, color="#F44336", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        exit_label = f"EXIT > ${eff_bp090:.1f}"
        if oi_adj_bp090 > 0 and next_bp090 > 0:
            exit_label += f" (原${next_bp090:.1f})"
        ax.annotate(exit_label,
                    xy=(last_xi, eff_bp090),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#F44336", fontweight="bold",
                    ha="left", va="center")

    # 当日标注
    if today is not None and today_close is not None and xi(today) is not None:
        ub_v = upper_band.get(today, 0)
        lb_v = lower_band.get(today, 0)
        bp_v = (today_close - lb_v) / (ub_v - lb_v) \
            if ub_v != lb_v else 0

        mc = {"BUY_CALL": "#2196F3", "SELL_PUT": "#FF9800",
              "EXIT": "#F44336"}.get(signal_type, "black")
        ax.scatter([xi(today)], [today_close], marker="D", s=120,
                   color=mc, edgecolors="black", linewidths=1.5, zorder=8)
        sl = {"BUY_CALL": "BUY CALL", "SELL_PUT": "SELL PUT",
              "EXIT": "EXIT"}.get(signal_type, "")
        ax.annotate(
            f"${today_close:.1f}  bp={bp_v:.2f}  RV={today_rv:.0%}"
            + (f"\n{sl}" if sl else ""),
            xy=(xi(today), today_close), xytext=(-60, -28),
            textcoords="offset points", fontsize=7.5, fontweight="bold",
            ha="center", color=mc,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mc,
                      alpha=0.9))

    # OI 关键价位
    if oi_levels is not None:
        mp = oi_levels.get("max_pain")
        cw = oi_levels.get("call_wall")
        pw = oi_levels.get("put_wall")
        if mp:
            ax.axhline(mp, color="orange", lw=1, ls=":", alpha=0.6, zorder=2)
            ax.annotate(f"Max Pain ${mp:.0f}", xy=(last_xi, mp),
                        xytext=(10, 6), textcoords="offset points",
                        fontsize=7, color="orange", fontweight="bold")
        if cw:
            ax.axhline(cw, color="red", lw=1, ls=":", alpha=0.5, zorder=2)
            ax.annotate(f"Call Wall ${cw:.0f}", xy=(last_xi, cw),
                        xytext=(10, 6), textcoords="offset points",
                        fontsize=7, color="red", fontweight="bold")
        if pw:
            ax.axhline(pw, color="green", lw=1, ls=":", alpha=0.5, zorder=2)
            ax.annotate(f"Put Wall ${pw:.0f}", xy=(last_xi, pw),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=7, color="green", fontweight="bold")

    # 格式
    parts = ["GLD 交易仪表板"]
    if today is not None:
        parts.append(today.strftime("%Y-%m-%d"))
        parts.append(f"Regime: {regime.get(today, '?')}")
        if signal_type:
            parts.append(f"信号: {signal_type.replace('_', ' ')}")
    ax.set_title("  |  ".join(parts), fontsize=13, fontweight="bold")
    ax.set_ylabel("GLD ($)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_xlim(-0.5, len(plot_dates) - 0.5)

    legend_el = [
        Line2D([0], [0], color="black", lw=1.5, label="GLD"),
        Line2D([0], [0], color="green", lw=1, alpha=0.6, label="Upper Band"),
        Line2D([0], [0], color="magenta", lw=1, alpha=0.6,
               label="Lower Band"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2196F3",
               markersize=9, label="Buy Call"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF9800",
               markersize=9, label="Sell Put"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#F44336",
               markersize=9, label="Exit"),
    ]
    if show_future and pred_u_pct is not None:
        legend_el.append(Line2D([0], [0], color="goldenrod", lw=1.2,
                                ls="--", label="5d Prediction"))
    if oi_adj_bands is not None and len(oi_adj_bands[0]) > 0:
        legend_el.append(Line2D([0], [0], color="darkgreen", lw=1.5,
                                ls="--", label="OI Adj Band"))
    if oi_levels is not None:
        legend_el.append(Line2D([0], [0], color="orange", lw=1, ls=":",
                                label="Max Pain"))
        legend_el.append(Line2D([0], [0], color="red", lw=1, ls=":",
                                label="Call Wall"))
        legend_el.append(Line2D([0], [0], color="green", lw=1, ls=":",
                                label="Put Wall"))
    ax.legend(handles=legend_el, loc="upper left", fontsize=7, ncol=4,
              framealpha=0.9)

    if trades:
        tdf = pd.DataFrame(trades)
        tg = tdf["gain"].mean()
        wr = (tdf["gain"] > 0).mean()
        hd = tdf["hold_days"].mean()
        n_bc = (tdf["sig_type"] == "BUY CALL").sum()
        n_sp = (tdf["sig_type"] == "SELL PUT").sum()
        summary = (f"BC({n_bc})+SP({n_sp})+Exit({len(ex_dates)}) | "
                   f"Avg:{tg:+.1f}% WR:{wr:.0%} Hold:{hd:.1f}d")
        ax.text(0.99, 0.02, summary, transform=ax.transAxes, fontsize=9,
                fontweight="bold", ha="right", va="bottom",
                bbox=dict(fc="lightyellow", ec="gray", alpha=0.85))

    plt.tight_layout()
    return fig, trades


def build_all_trades(close, high, bp_dates, buy_call, sell_put, exit_sig,
                     start_date=None, max_hold=10):
    """构建指定时间范围内的全部交易 (不重叠)."""
    all_dates = close.index
    trades = []
    in_trade = False
    entry_date = entry_price = sig_type = None
    peak = 0

    scan_dates = bp_dates[bp_dates >= start_date] if start_date else bp_dates

    for d in scan_dates:
        if not in_trade:
            if buy_call.get(d, False):
                in_trade = True
                entry_date, sig_type, entry_price = d, "BUY CALL", close[d]
                peak = entry_price
                hold = 0
            elif sell_put.get(d, False):
                in_trade = True
                entry_date, sig_type, entry_price = d, "SELL PUT", close[d]
                peak = entry_price
                hold = 0
        else:
            hold += 1
            fc = close.get(d, entry_price)
            fh = high.get(d, fc)
            peak = max(peak, fh)

            should_exit = False
            exit_type = "Timeout"

            if exit_sig.get(d, False):
                should_exit, exit_type = True, "BandExit"
            else:
                ppct = (peak / entry_price - 1) * 100
                dd = (peak - fc) / peak * 100
                if ppct > 2.0 and dd >= 1.5:
                    should_exit, exit_type = True, "Pullback"

            if hold >= max_hold:
                should_exit = True

            if should_exit:
                exit_price = fc
                g = (exit_price / entry_price - 1) * 100
                trades.append(dict(
                    entry_date=entry_date, exit_date=d,
                    sig_type=sig_type, exit_type=exit_type,
                    entry_price=entry_price, exit_price=exit_price,
                    gain=g, hold_days=hold))
                in_trade = False

    return trades


def _build_nav(trades, period_dates):
    """从交易列表构建净值曲线."""
    nav = pd.Series(np.nan, index=period_dates)
    nav.iloc[0] = 100.0
    cur = 100.0
    for t in trades:
        ed, xd = t["entry_date"], t["exit_date"]
        g = t["gain"] / 100
        if ed in nav.index:
            nav[ed] = cur
        cur *= (1 + g)
        if xd in nav.index:
            nav[xd] = cur
    return nav.ffill().fillna(100.0)


def _trade_stats(trades, buy_hold):
    """计算交易统计."""
    if not trades:
        return {}
    tdf = pd.DataFrame(trades)
    cum = 1.0
    for g in tdf["gain"]:
        cum *= (1 + g / 100)
    total_ret = (cum - 1) * 100
    bh_ret = (buy_hold.iloc[-1] / 100 - 1) * 100
    nav = _build_nav(trades, buy_hold.index)
    running_max = nav.cummax()
    max_dd = ((nav - running_max) / running_max * 100).min()
    return {
        "n": len(tdf), "wr": (tdf["gain"] > 0).mean(),
        "avg": tdf["gain"].mean(), "total": total_ret,
        "bh": bh_ret, "max_dd": max_dd,
        "hold": tdf["hold_days"].mean(),
        "max_g": tdf["gain"].max(), "min_g": tdf["gain"].min(),
    }


def generate_backtest_chart(close, high, low, bp_dates, upper_band,
                            lower_band, buy_call, sell_put, exit_sig,
                            regime, rv_pctile, gld_1h=None):
    """v2.2 回测: v1.0(收盘) vs v2.2(H/L入场+12h止盈), 1Y/2Y/3Y."""
    from core.signals_v2 import run_backtest as run_v22

    last = bp_dates[-1]
    periods = [
        ("近6月", last - pd.DateOffset(months=6)),
        ("近1年", last - pd.DateOffset(years=1)),
        ("近2年", last - pd.DateOffset(years=2)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)
    fig.suptitle("回测对比: v1.0 收盘价 vs v2.2 盘中H/L+12h止盈",
                 fontsize=15, fontweight="bold")

    summary_rows = []

    for ax, (label, start) in zip(axes, periods):
        period_dates = close.index[close.index >= start]
        if len(period_dates) == 0:
            ax.set_title(f"{label}: 无数据")
            continue

        buy_hold = pd.Series(index=period_dates, dtype=float)
        base = close.get(period_dates[0], close.iloc[0])
        for d in period_dates:
            buy_hold[d] = close.get(d, base) / base * 100

        # v1.0 回测 (收盘价)
        trades_v1 = build_all_trades(close, high, bp_dates,
                                      buy_call, sell_put, exit_sig,
                                      start_date=start)
        # v2.2 回测 (H/L入场 + 12h止盈)
        trades_v2 = run_v22(close, high, low, upper_band, lower_band,
                            regime, rv_pctile, gld_1h=gld_1h,
                            start_date=start)

        nav_v1 = _build_nav(trades_v1, period_dates)
        nav_v2 = _build_nav(trades_v2, period_dates)
        s1 = _trade_stats(trades_v1, buy_hold)
        s2 = _trade_stats(trades_v2, buy_hold)

        # 画图
        ax.plot(buy_hold.index, buy_hold.values, color="gray", lw=1.2,
                alpha=0.5, label="买入持有")
        if s1:
            ax.plot(nav_v1.index, nav_v1.values, color="#2196F3", lw=1.5,
                    label=f"v1.0 收盘 ({s1['n']}笔 {s1['total']:+.1f}%)")
        if s2:
            ax.plot(nav_v2.index, nav_v2.values, color="#FF9800", lw=2,
                    label=f"v2.2 盘中 ({s2['n']}笔 {s2['total']:+.1f}%)")

        # 交易标注 (v2.2)
        for t in trades_v2:
            ce = "#FF9800"
            if t["entry_date"] in nav_v2.index:
                ax.scatter([t["entry_date"]], [nav_v2[t["entry_date"]]],
                           marker="^", s=50, color=ce, edgecolors="black",
                           linewidths=0.4, zorder=5)

        # Regime 背景
        reg = regime.reindex(period_dates)
        bull = reg == "Bull"
        if bull.any():
            starts_b = period_dates[bull & (~bull.shift(1, fill_value=False))]
            ends_b = period_dates[bull & (~bull.shift(-1, fill_value=False))]
            for s, e in zip(starts_b, ends_b):
                ax.axvspan(s, e, alpha=0.03, color="green")

        ax.axhline(100, color="black", lw=0.5, ls=":", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("净值 (起始=100)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # 统计文本
        if s2:
            stats_text = (
                f"v2.2: {s2['total']:+.1f}% ({s2['n']}笔 {s2['wr']:.0%}WR "
                f"均{s2['avg']:+.1f}% 回撤{s2['max_dd']:.1f}% "
                f"持仓{s2['hold']:.1f}d)"
                f"  |  v1.0: {s1['total']:+.1f}% ({s1['n']}笔)"
                f"  |  持有: {s2.get('bh', 0):+.1f}%")
        elif s1:
            stats_text = f"v1.0: {s1['total']:+.1f}% ({s1['n']}笔) | 持有: {s1['bh']:+.1f}%"
        else:
            stats_text = "无交易"

        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(fc="lightyellow", ec="gray", alpha=0.9))
        ax.set_title(f"{label} ({start.strftime('%Y-%m')} ~ "
                     f"{last.strftime('%Y-%m')})", fontsize=12,
                     fontweight="bold")
        ax.set_ylabel("净值 (起始=100)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.axhline(100, color="black", lw=0.5, ls=":", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        bh_running_max = buy_hold.cummax()
        bh_max_dd = ((buy_hold - bh_running_max) / bh_running_max * 100).min()

        row = {"周期": label, "买入持有": f"{s2.get('bh', s1.get('bh', 0)):+.1f}%",
               "持有回撤": f"{bh_max_dd:.1f}%"}
        for tag, s in [("v1.0", s1), ("v2.2", s2)]:
            if s:
                row[f"{tag}收益"] = f"{s['total']:+.1f}%"
                row[f"{tag}交易"] = s["n"]
                row[f"{tag}胜率"] = f"{s['wr']:.0%}"
                row[f"{tag}回撤"] = f"{s['max_dd']:.1f}%"
        summary_rows.append(row)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, summary_rows


def compute_next_day_band(close, range_df, bp_dates, today):
    """计算下一交易日 band 和信号阈值."""
    today_close_val = close.get(today, 0)
    if today not in range_df.index:
        return 0, 0, 0, 0

    pu = range_df.loc[today, "pred_upper_pct"]
    next_upper = today_close_val * (1 + pu / 100)

    lowers = []
    today_loc = bp_dates.get_loc(today) if today in bp_dates else -1
    for offset in range(3):
        idx = today_loc - offset
        if idx < 0:
            break
        d = bp_dates[idx]
        if d in range_df.index:
            c = close.get(d, 0)
            pl = range_df.loc[d, "pred_lower_pct"]
            lowers.append(c * (1 + pl / 100))
    next_lower = np.mean(lowers) if lowers else 0

    if next_upper <= next_lower:
        return next_upper, next_lower, 0, 0
    bp030 = next_lower + 0.30 * (next_upper - next_lower)
    bp090 = next_lower + 0.90 * (next_upper - next_lower)
    return next_upper, next_lower, bp030, bp090


# ══════════════════════════════════════════════════════════
# v2.2 盘中信号模式
# ══════════════════════════════════════════════════════════
def _render_intraday_mode(close_d, high_d, low_d, upper_band, lower_band,
                          regime, rv_pctile, bp_dates, bp_s,
                          gc_gld_ratio, usdcny_rate, today_sgt):
    """v2.2: v1.0 Band + 盘中H/L入场 + 12h止盈."""
    from core.signals_v2 import (generate_daily_signals, run_backtest,
                                  EXIT_TIMEFRAME, PULLBACK_GAIN, PULLBACK_DD)

    # 时区
    tz_name = st.sidebar.selectbox("时区", list(TZ_OPTIONS.keys()),
                                    index=list(TZ_OPTIONS.keys()).index(TZ_DEFAULT))

    # 自动刷新
    if _HAS_AUTOREFRESH:
        refresh_min = st.sidebar.selectbox("自动刷新", [0, 1, 3, 5, 10],
                                            index=3, format_func=lambda x: "关闭" if x == 0 else f"{x}分钟")
        if refresh_min > 0:
            st_autorefresh(interval=refresh_min * 60 * 1000,
                           key="intraday_refresh")

    # GLD 1h
    gld_1h_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "Gold", "data", "raw", "market", "gld_1h.csv")
    gld_1h_path = os.path.normpath(gld_1h_path)
    gld_1h = pd.read_csv(gld_1h_path, index_col=0, parse_dates=True) \
        if os.path.exists(gld_1h_path) else None

    # 信号 (同 v1.0 的 Band, H/L 触发)
    sig_df = generate_daily_signals(
        close_d, high_d, low_d, upper_band, lower_band,
        regime, rv_pctile)

    # 回测 (12h 止盈)
    trades = run_backtest(
        close_d, high_d, low_d, upper_band, lower_band,
        regime, rv_pctile, gld_1h=gld_1h,
        start_date=pd.Timestamp(today_sgt) - timedelta(days=180))

    # 下一交易日阈值 (与 v1.0 完全一致)
    from core.data import load_config, load_oos_predictions
    range_df = load_oos_predictions(load_config())
    last_date = bp_dates[-1]
    last_close = close_d.get(last_date, 0)
    last_regime = regime.get(last_date, "?")
    last_bp = bp_s.get(last_date, 0)
    next_upper, next_lower, next_bp030, next_bp090 = \
        compute_next_day_band(close_d, range_df, bp_dates, last_date)

    # OI 微观结构修正 (期权到期压缩效应)
    oi_adj_bp030 = oi_adj_bp090 = 0
    _cfg_oi = load_config()
    _eod_oi, _snap_oi = load_latest_eod_snapshot(_cfg_oi)
    if _eod_oi is not None and next_upper > next_lower:
        _oi = compute_oi_factors(_eod_oi, last_close, ref_date=today_sgt)
        if _oi is not None:
            adj_u, adj_l, _oi_det = adjust_range(
                next_upper, next_lower, last_close, _oi)
            if adj_u > adj_l:
                oi_adj_bp030 = adj_l + 0.30 * (adj_u - adj_l)
                oi_adj_bp090 = adj_l + 0.90 * (adj_u - adj_l)

    # 使用 OI 修正后的阈值 (如有)
    eff_bp030 = oi_adj_bp030 if oi_adj_bp030 > 0 else next_bp030
    eff_bp090 = oi_adj_bp090 if oi_adj_bp090 > 0 else next_bp090

    gc_gld_r = gc_gld_ratio if gc_gld_ratio else 10.9
    rt = _get_realtime_prices()
    _cny = rt["usdcny"] if rt else (usdcny_rate if usdcny_rate else 7.0)
    _g = 31.1035

    # 判断是否有未平仓: 看最后一笔回测交易
    has_open_position = False
    entry_price_open = peak_open = pullback_stop = 0
    if trades:
        last_trade = trades[-1]
        # 最后一笔已平仓 → 检查之后是否有新买入信号
        buy_after_last = sig_df[
            (sig_df["buy_signal"]) &
            (sig_df.index > last_trade["exit_date"])
        ]
        if len(buy_after_last) > 0:
            last_buy = buy_after_last.index[-1]
            has_open_position = True
            entry_price_open = buy_after_last.loc[last_buy, "bp030_price"]
            post_entry = high_d[high_d.index >= last_buy]
            peak_open = post_entry.max() if len(post_entry) > 0 else entry_price_open
            gain_pct = (peak_open / entry_price_open - 1) * 100
            if gain_pct > PULLBACK_GAIN:
                pullback_stop = peak_open * (1 - PULLBACK_DD / 100)

    # ════════════════════════════════════════
    # 醒目顶部: 交易价位 + 实时
    # ════════════════════════════════════════
    if eff_bp030 > 0:
        # 实时价格
        gc_now = rt["gc_price"] if rt else 0
        gld_est = gc_now / gc_gld_r if gc_now > 0 else last_close
        bp_est = (gld_est - next_lower) / (next_upper - next_lower) \
            if next_upper > next_lower else 0
        zone = "买入区" if bp_est < 0.30 else ("退出区" if bp_est > 0.90 else "观望")
        zone_icon = {"买入区": "🟢", "退出区": "🔴", "观望": "⚪"}
        ts = rt["timestamp"] if rt else ""

        # 信号预测价位 (蓝色背景)
        st.markdown("""<style>
        .signal-box {background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
                     border-radius: 10px; padding: 15px; margin-bottom: 10px;
                     border-left: 4px solid #1565C0;}
        .price-box  {background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
                     border-radius: 10px; padding: 15px;
                     border-left: 4px solid #7B1FA2;}
        </style>""", unsafe_allow_html=True)

        st.markdown('<div class="signal-box">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            oi_tag = " (OI修正)" if oi_adj_bp030 > 0 else ""
            st.metric(f"买入 <{oi_tag}", f"${eff_bp030:.2f}",
                      delta=f"COMEX < ${eff_bp030*gc_gld_r:.0f} | 沪金 < ¥{eff_bp030*gc_gld_r*_cny/_g:.1f}")
        with c2:
            st.metric(f"退出 >{oi_tag}", f"${eff_bp090:.2f}",
                      delta=f"COMEX > ${eff_bp090*gc_gld_r:.0f} | 沪金 > ¥{eff_bp090*gc_gld_r*_cny/_g:.1f}")
        with c3:
            sig_text = sig_df.loc[last_date]["signal_text"] \
                if last_date in sig_df.index else ""
            st.metric("最新信号", sig_text if sig_text else "—",
                      delta=f"Regime: {last_regime} | bp={last_bp:.3f} | RV={rv_pctile.get(last_date,0):.0%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 实时价格行 (紫色背景)
        if gc_now > 0:
            xau_est = gc_now
            shfe_est = gc_now * _cny / _g

            st.markdown('<div class="price-box">', unsafe_allow_html=True)
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                st.metric("COMEX 纽约金", f"${gc_now:.1f}",
                          delta=f"{zone_icon.get(zone,'')} {zone}")
            with r2:
                st.metric("伦敦金 XAU", f"${xau_est:.1f}",
                          delta="≈COMEX")
            with r3:
                st.metric("GLD", f"${gld_est:.1f}",
                          delta=f"bp≈{bp_est:.2f}")
            with r4:
                st.metric("沪金 AU", f"¥{shfe_est:.2f}",
                          delta=f"USD/CNY={_cny:.4f}")
            with r5:
                st.metric("数据时间", ts if ts else "—",
                          delta=f"数据: {last_date.date()} | 今日: {today_sgt}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption(f"实时数据未获取 | GLD 收盘 ${last_close:.2f} ({last_date.date()})")

    # ── 信号历史图 ──
    st.divider()
    lookback_days = st.sidebar.slider("回看天数", 30, 180, 65)
    lookback = last_date - timedelta(days=lookback_days)
    viz_dates = close_d.index[(close_d.index >= lookback) & (close_d.index <= last_date)]
    sig_viz = sig_df.reindex(viz_dates).dropna(subset=["close"])

    fig, ax = plt.subplots(figsize=(18, 9))

    # index-based x-axis
    plot_dates = list(viz_dates)
    d2i = {d: i for i, d in enumerate(plot_dates)}
    def xi(d): return d2i.get(d)
    def xi_arr(dates): return [d2i[d] for d in dates if d in d2i]
    def _fmt_tick(x, pos):
        idx = int(round(x))
        if 0 <= idx < len(plot_dates):
            return plot_dates[idx].strftime("%m/%d")
        return ""

    # 价格 + H/L 范围
    cl_plot = close_d.reindex(viz_dates).dropna()
    ax.plot(xi_arr(cl_plot.index), cl_plot.values, "k-", lw=1.8, zorder=3)
    hi_plot = high_d.reindex(viz_dates).dropna()
    lo_plot = low_d.reindex(viz_dates).dropna()
    hl_common = hi_plot.index.intersection(lo_plot.index)
    if len(hl_common) > 0:
        ax.fill_between(xi_arr(hl_common), lo_plot[hl_common].values,
                         hi_plot[hl_common].values, alpha=0.08, color="gray")

    # Band
    ub_plot = upper_band.reindex(viz_dates).dropna()
    lb_plot = lower_band.reindex(viz_dates).dropna()
    cidx = ub_plot.index.intersection(lb_plot.index)
    if len(cidx) > 0:
        ax.fill_between(xi_arr(cidx), lb_plot[cidx].values,
                         ub_plot[cidx].values, alpha=0.06, color="green")
        ax.plot(xi_arr(cidx), ub_plot[cidx].values, color="green", lw=1, alpha=0.5)
        ax.plot(xi_arr(cidx), lb_plot[cidx].values, color="magenta", lw=1, alpha=0.5)
        bp030_line = lb_plot[cidx] + 0.30 * (ub_plot[cidx] - lb_plot[cidx])
        bp090_line = lb_plot[cidx] + 0.90 * (ub_plot[cidx] - lb_plot[cidx])
        ax.plot(xi_arr(cidx), bp030_line.values, color="#2196F3", lw=0.8, ls="--", alpha=0.5)
        ax.plot(xi_arr(cidx), bp090_line.values, color="#F44336", lw=0.8, ls="--", alpha=0.5)

    # 信号标注 (每天一个标记, 在触发价位上)
    buy_days = sig_viz[sig_viz["buy_signal"]]
    exit_days = sig_viz[sig_viz["exit_signal"]]

    for d, r in buy_days.iterrows():
        if xi(d) is None:
            continue
        color = "#2196F3" if r["buy_type"] == "BUY CALL" else "#FF9800"
        ax.scatter([xi(d)], [r["bp030_price"]], marker="^", s=120, color=color,
                   edgecolors="black", lw=0.7, zorder=6)

    for d, r in exit_days.iterrows():
        if xi(d) is None:
            continue
        ax.scatter([xi(d)], [r["bp090_price"]], marker="v", s=100,
                   color="#F44336", edgecolors="black", lw=0.7, zorder=5)

    # 回测止盈标注 (淡色, 每天一个)
    tdf_viz = pd.DataFrame(trades) if trades else pd.DataFrame()
    if len(tdf_viz) > 0:
        for _, t in tdf_viz.iterrows():
            xd = t["exit_date"]
            if xd not in d2i or t["exit_type"] == "BandExit":
                continue
            cx = {"Pullback": "#FF6600", "MACD": "#9C27B0", "Timeout": "gray"}
            mk = {"Pullback": "s", "MACD": "D", "Timeout": "X"}
            ax.scatter([xi(xd)], [t["exit_price"]],
                       marker=mk.get(t["exit_type"], "o"), s=80,
                       color=cx.get(t["exit_type"], "gray"),
                       edgecolors="black", lw=0.5, alpha=0.5, zorder=4)
            ax.annotate(f"{t['gain']:+.1f}%", xy=(xi(xd), t["exit_price"]),
                        xytext=(3, 6), textcoords="offset points", fontsize=6,
                        color=cx.get(t["exit_type"], "gray"), alpha=0.7)

    legend_el = [
        Line2D([0],[0], color="k", lw=1.5, label="GLD Close"),
        Line2D([0],[0], color="green", lw=1, alpha=0.5, label="Band"),
        Line2D([0],[0], color="#2196F3", lw=0.8, ls="--", label="Buy bp=0.30"),
        Line2D([0],[0], color="#F44336", lw=0.8, ls="--", label="Exit bp=0.90"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#2196F3", markersize=9, label="BUY CALL"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#FF9800", markersize=9, label="SELL PUT"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#F44336", markersize=9, label="BandExit"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#FF6600", markersize=8, alpha=0.5, label="Pullback"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#9C27B0", markersize=8, alpha=0.5, label="MACD止盈"),
    ]
    ax.legend(handles=legend_el, loc="upper left", fontsize=6, ncol=5)
    ax.set_title(f"v2.2 盘中信号 (v1.0 Band + H/L入场 + {EXIT_TIMEFRAME}止盈) | "
                 f"数据至 {last_date.date()} | Regime: {last_regime}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("GLD ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    st.download_button("下载图表", buf.getvalue(),
                       file_name="gld_v21_dashboard.png", mime="image/png")
    plt.close(fig)

    # ── 期权策略预判 ──
    st.divider()
    st.subheader("期权策略推荐")
    _cfg_opt2 = load_config()
    _eod_opt2, _snap_opt2 = load_latest_eod_snapshot(_cfg_opt2)
    _sig_now2 = None
    if last_date in sig_df.index:
        _r2 = sig_df.loc[last_date]
        if _r2["buy_signal"]:
            _sig_now2 = _r2["buy_type"].replace(" ", "_") if _r2["buy_type"] else "BUY_CALL"
        if _r2["exit_signal"]:
            _sig_now2 = _sig_now2 or "EXIT"
    _render_options_section(_eod_opt2, _snap_opt2, last_close, eff_bp090,
                            oi_adj_bp090=oi_adj_bp090,
                            gc_gld_ratio=gc_gld_ratio,
                            today_sgt=today_sgt, current_signal=_sig_now2)

    # ── 止盈预测 ──
    st.divider()
    st.subheader("止盈预测")

    # 用回测结果判断是否已平仓 (包含 Pullback/MACD, 不只是 BandExit)
    trade_exits = {}  # {entry_date: (exit_date, exit_type, gain)}
    for t in trades:
        trade_exits[t["entry_date"]] = (t["exit_date"], t["exit_type"], t["gain"])

    buy_rows = sig_df[sig_df["buy_signal"]]
    tp_recs = []

    for buy_d in buy_rows.index[-5:][::-1]:
        br = buy_rows.loc[buy_d]
        entry_p = br["bp030_price"]

        # 从入场到现在的峰值
        post = high_d[(high_d.index >= buy_d) & (high_d.index <= last_date)]
        pk = post.max() if len(post) > 0 else entry_p
        gain = (pk / entry_p - 1) * 100
        current_p = close_d.get(last_date, entry_p)
        current_gain = (current_p / entry_p - 1) * 100

        # Pullback 止盈位
        pb_stop = pk * (1 - PULLBACK_DD / 100) if gain > PULLBACK_GAIN else 0
        band_stop = eff_bp090

        # 判断是否已平仓: 优先看回测, 其次看 BandExit 信号
        if buy_d in trade_exits:
            xd, xt, xg = trade_exits[buy_d]
            status = f"已{xt} {xd.strftime('%m/%d')} {xg:+.1f}%"
        else:
            status = "持仓中"

        tp_recs.append({
            "买入日": buy_d.strftime("%m/%d"),
            "类型": br["buy_type"],
            "入场价": f"${entry_p:.1f}",
            "峰值": f"${pk:.1f} ({gain:+.1f}%)",
            "当前": f"${current_p:.1f} ({current_gain:+.1f}%)",
            "Pullback止盈": f"${pb_stop:.1f}" if pb_stop > 0 else f"未达{PULLBACK_GAIN}%",
            "BandExit": f"${band_stop:.1f}",
            "状态": status,
        })

    if tp_recs:
        st.dataframe(pd.DataFrame(tp_recs), use_container_width=True, hide_index=True)
    else:
        st.caption("无近期买入信号")

    # ── 信号历史表 ──
    st.divider()
    st.subheader("近期信号")
    sig_recent = sig_df[sig_df["signal_text"] != ""].tail(15)
    if len(sig_recent) > 0:
        recs = []
        for d, r in sig_recent.iterrows():
            recs.append({
                "日期": d.strftime("%Y-%m-%d"),
                "GLD": f"${r['close']:.2f}",
                "H/L": f"${r['low']:.1f}~${r['high']:.1f}",
                "bp(L/C/H)": f"{r['bp_low']:.2f}/{r['bp_close']:.2f}/{r['bp_high']:.2f}",
                "买入价": f"${r['bp030_price']:.2f}",
                "退出价": f"${r['bp090_price']:.2f}",
                "信号": r["signal_text"],
            })
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)

    # ── 12h 止盈回测 ──
    st.divider()
    st.subheader("近期交易回测 (12h止盈)")
    trades = run_backtest(
        close_d, high_d, low_d, upper_band, lower_band,
        regime, rv_pctile, gld_1h=gld_1h,
        start_date=pd.Timestamp(today_sgt) - timedelta(days=90))
    if trades:
        tdf = pd.DataFrame(trades)
        total_ret = ((1 + tdf["gain"] / 100).prod() - 1) * 100
        wr = (tdf["gain"] > 0).mean()
        st.markdown(f"**{len(trades)}笔 | 胜率{wr:.0%} | 累计{total_ret:+.1f}% | "
                    f"均持仓{tdf['hold_days'].mean():.1f}d**")
        trecs = []
        for _, t in tdf.iterrows():
            trecs.append({
                "入场": t["entry_date"].strftime("%m/%d"),
                "类型": t["type"],
                "入场价": f"${t['entry_price']:.1f}",
                "出场": t["exit_date"].strftime("%m/%d"),
                "退出": t["exit_type"],
                "出场价": f"${t['exit_price']:.1f}",
                "收益": f"{t['gain']:+.1f}%",
                "持仓": f"{t['hold_days']}d",
            })
        st.dataframe(pd.DataFrame(trecs), use_container_width=True, hide_index=True)
    else:
        st.info("近3个月无交易")

    # ── 模型信息 ──
    with st.expander("模型信息"):
        from core.signals_v2 import EXIT_TIMEFRAME, PULLBACK_GAIN, PULLBACK_DD
        gld_1h_info = f"{gld_1h.index[0].strftime('%Y-%m-%d')} ~ {gld_1h.index[-1].strftime('%Y-%m-%d')} ({len(gld_1h)} bars)" \
            if gld_1h is not None else "未加载"
        st.markdown(f"""
- **Band**: v1.0 日线模型 (20年训练, LSTM+Attention, Conformal 80%覆盖)
- **入场**: 日线 Low 触及 bp<0.30 即入场 (盘中触发, 不等收盘)
- **止盈尺度**: **{EXIT_TIMEFRAME}** (可配置: 1h/2h/4h/8h/12h)
- **退出优先级**: BandExit (bp>0.90) > Pullback (涨>{PULLBACK_GAIN}%回撤>{PULLBACK_DD}%) > MACD弱化 > Timeout (10d)
- **GLD 1h 数据**: {gld_1h_info}
- **持仓周期**: 2-5天 (适合期权)
- **回测 (2025-09~2026-03)**: 13笔 85%胜率 +37.5%累计 Sharpe=0.78
""")


# ══════════════════════════════════════════════════════════
# 共享: 期权策略推荐
# ══════════════════════════════════════════════════════════
def _render_options_section(eod_df, snap_date, last_close, next_bp090,
                            oi_adj_bp090=0, gc_gld_ratio=None,
                            today_sgt=None, current_signal=None):
    """渲染期权策略推荐 (盘中信号 + 今日预测 共用).

    始终显示 BUY CALL 和 SELL PUT 两种预判, 不等信号触发.
    """
    if eod_df is None:
        cfg = load_config()
        eod_df, snap_date = load_latest_eod_snapshot(cfg)

    if eod_df is None:
        st.info("无期权快照数据")
        return

    eff_exit = oi_adj_bp090 if oi_adj_bp090 > 0 else next_bp090
    _rt = _get_realtime_prices()
    _gc_gld_r = gc_gld_ratio if gc_gld_ratio else 10.9
    current_gld = _rt["gc_price"] / _gc_gld_r if _rt else last_close
    price_src = f"实时 GLD≈${current_gld:.1f}" if _rt else f"收盘 ${last_close:.2f}"

    # 当前信号状态提示
    if current_signal == "EXIT":
        st.warning("EXIT 信号活跃 — 建议平仓现有头寸")
    elif current_signal in ("BUY_CALL", "SELL_PUT"):
        st.success(f"{current_signal.replace('_', ' ')} 信号活跃 — 以下策略可立即执行")
    else:
        st.info("当前观望区 — 以下为预判策略, 信号触发时可立即执行")

    # BUY CALL 策略
    result_call = get_strategy_table("BUY_CALL", current_gld, eff_exit, eod_df,
                                      use_live=True)
    data_src = result_call.get("source", "EOD")

    st.caption(f"期权数据: **{data_src}** | 当前价: {price_src} | "
               f"退出目标: ${eff_exit:.2f}")

    if result_call.get("rec"):
        st.info(result_call["rec"])

    st.markdown("**单腿 Call (Long Call)**")
    if result_call.get("single_leg"):
        st.dataframe(pd.DataFrame(result_call["single_leg"]),
                     use_container_width=True, hide_index=True)

    st.markdown("**牛市看涨价差 (Bull Call Spread)**")
    if result_call.get("spread"):
        st.dataframe(pd.DataFrame(result_call["spread"]),
                     use_container_width=True, hide_index=True)

    # SELL PUT 策略
    result_put = get_strategy_table("SELL_PUT", current_gld, eff_exit, eod_df,
                                     use_live=True)
    st.markdown("**牛市看跌价差 (Bull Put Spread)** — 高IV时推荐")
    if result_put.get("spread"):
        st.dataframe(pd.DataFrame(result_put["spread"]),
                     use_container_width=True, hide_index=True)

    st.caption("风控: 仓位2-5%(稳健)/5-10%(中性)/≤5%(激进) | "
               "平仓: bp>0.90 / Pullback / MACD弱化 / 10d")


# ══════════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════════
def main():
    today_sgt = get_today_sgt()
    st.title(f"GLD 期权交易仪表板  ({today_sgt})")

    # 自动检测并更新市场数据
    cfg_refresh = load_config()
    with st.spinner("检测数据更新..."):
        refresh_results = auto_refresh_market_data(cfg_refresh)
        refreshed = [f"{t}: {s}" for t, s in refresh_results if "更新" in s]

        # 增量更新特征 + 扩展 OOS 预测
        try:
            n_feat = update_features_incremental(cfg_refresh)
            if n_feat > 0:
                refresh_results.append(("特征", f"+{n_feat}天"))
            n_new, oos_msg = extend_oos_predictions(cfg_refresh)
            refresh_results.append(("OOS预测", oos_msg))
            if n_new > 0:
                refreshed.append(f"OOS: {oos_msg}")
        except Exception as e:
            refresh_results.append(("OOS预测", f"失败: {e}"))

        if refreshed:
            load_all.clear()
            st.toast("数据已更新: " + " | ".join(refreshed), icon="✅")

    with st.spinner("加载数据..."):
        gld, range_df, regime, rv_pctile, gc_gld_ratio, usdcny_rate = load_all()

    close, high, low = gld["Close"], gld["High"], gld["Low"]

    # 信号计算
    upper_band, lower_band, bp = build_band(
        range_df, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp_dates = bp.dropna().index
    bp_s = bp.reindex(bp_dates)
    rv_p = rv_pctile.reindex(bp_dates)
    is_bull = regime.reindex(bp_dates) == "Bull"
    buy_call, sell_put, exit_sig = generate_signals(bp_s, rv_p, is_bull)

    last_date = bp_dates[-1]
    last_close = close.get(last_date, 0)
    last_bp = bp_s.get(last_date, 0)
    last_regime = regime.get(last_date, "?")
    last_rv = rv_p.get(last_date, 0)

    # ── 侧边栏 ──
    st.sidebar.header("设置")
    mode = st.sidebar.radio("模式", ["盘中信号", "今日预测", "历史回看", "回测分析"])

    # 数据状态
    with st.sidebar.expander("数据状态", expanded=False):
        st.caption(f"今日 (SGT): {today_sgt}")
        st.caption(f"GLD 最新: {last_date.date()}")
        for t, s in refresh_results:
            st.caption(f"{t}: {s}")

    if mode == "回测分析":
        # ── 回测模式 ──
        st.divider()
        st.subheader("回测对比: v1.0 收盘 vs v2.2 盘中+12h止盈")

        # 加载 GLD 1h (用于 v2.2 止盈)
        _1h_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Gold", "data", "raw", "market", "gld_1h.csv")
        _1h_path = os.path.normpath(_1h_path)
        _gld_1h = pd.read_csv(_1h_path, index_col=0, parse_dates=True) \
            if os.path.exists(_1h_path) else None

        bt_fig, bt_summary = generate_backtest_chart(
            close, high, low, bp_dates, upper_band, lower_band,
            buy_call, sell_put, exit_sig,
            regime, rv_pctile, gld_1h=_gld_1h)
        st.pyplot(bt_fig, use_container_width=True)

        if bt_summary:
            st.subheader("回测统计")
            st.dataframe(pd.DataFrame(bt_summary),
                         use_container_width=True, hide_index=True)

        import io
        buf = io.BytesIO()
        bt_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
        buf.seek(0)
        st.download_button("下载回测图", buf.getvalue(),
                           file_name="backtest.png", mime="image/png")
        plt.close(bt_fig)

        st.caption("注: 回测基于标的(GLD)价格变化, 非期权实际损益. "
                   "期权杠杆效应会放大实际收益/亏损.")
        return  # 回测模式不显示其他内容

    if mode == "盘中信号":
        _render_intraday_mode(close, high, low, upper_band, lower_band,
                              regime, rv_pctile, bp_dates, bp_s,
                              gc_gld_ratio, usdcny_rate, today_sgt)
        return

    if mode == "历史回看":
        min_d = bp_dates[0].to_pydatetime().date()
        max_d = bp_dates[-1].to_pydatetime().date()
        c1, c2 = st.sidebar.columns(2)
        presets = {"近2月": 65, "近半年": 180, "近1年": 365,
                   "近2年": 730, "近5年": 1825, "全部": 9999}
        preset = st.sidebar.selectbox("快速选择", list(presets.keys()))
        default_start = max_d - timedelta(days=presets[preset])
        if default_start < min_d:
            default_start = min_d
        with c1:
            start_date = st.date_input("开始", value=default_start,
                                       min_value=min_d, max_value=max_d)
        with c2:
            end_date = st.date_input("结束", value=max_d,
                                     min_value=min_d, max_value=max_d)

        viz_dates = close.index[
            (close.index >= pd.Timestamp(start_date)) &
            (close.index <= pd.Timestamp(end_date))]
        show_future = False
        pred_u_pct = pred_l_pct = None
        next_bp030 = next_bp090 = 0
        ref_date = pd.Timestamp(end_date)
        today_for_chart = ref_date if ref_date in bp_dates else None
        today_close_chart = close.get(today_for_chart, 0) \
            if today_for_chart else 0
        today_rv_chart = rv_pctile.get(today_for_chart, 0) \
            if today_for_chart else 0

        # 长周期 (>120天) 不显示交易信号, 用空信号
        is_long_range = len(viz_dates) > 120
        sig_type_viz = None
        if not is_long_range and today_for_chart is not None:
            if buy_call.get(today_for_chart, False):
                sig_type_viz = "BUY_CALL"
            elif sell_put.get(today_for_chart, False):
                sig_type_viz = "SELL_PUT"
            if exit_sig.get(today_for_chart, False):
                sig_type_viz = sig_type_viz or "EXIT"

        # 长周期用空信号 dict 禁用信号标注
        if is_long_range:
            buy_call_viz = {}
            sell_put_viz = {}
            exit_sig_viz = {}
        else:
            buy_call_viz = buy_call
            sell_put_viz = sell_put
            exit_sig_viz = exit_sig
    else:
        lookback_days = st.sidebar.slider("回看天数", 30, 180, 65)
        lookback = last_date - timedelta(days=lookback_days)
        viz_dates = close.index[
            (close.index >= lookback) & (close.index <= last_date)]

        pred_u_pct = range_df.loc[last_date, "pred_upper_pct"] \
            if last_date in range_df.index else 0
        pred_l_pct = range_df.loc[last_date, "pred_lower_pct"] \
            if last_date in range_df.index else 0
        show_future = True
        today_for_chart = last_date
        today_close_chart = last_close
        today_rv_chart = last_rv

        sig_type_viz = None
        if buy_call.get(last_date, False):
            sig_type_viz = "BUY_CALL"
        elif sell_put.get(last_date, False):
            sig_type_viz = "SELL_PUT"
        if exit_sig.get(last_date, False):
            sig_type_viz = sig_type_viz or "EXIT"

        next_upper, next_lower, next_bp030, next_bp090 = \
            compute_next_day_band(close, range_df, bp_dates, last_date)

    # ── 顶部指标 ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        delta_pct = None
        if len(close) > 1:
            delta_pct = f"{(last_close / close.iloc[-2] - 1) * 100:+.2f}%"
        st.metric("GLD", f"${last_close:.2f}", delta=delta_pct)
    with m2:
        st.metric("Regime", last_regime)
    with m3:
        st.metric("Band Position", f"{last_bp:.3f}")
    with m4:
        st.metric("RV Percentile", f"{last_rv:.0%}")
    with m5:
        sig_disp = sig_type_viz.replace("_", " ") if sig_type_viz else "无信号"
        icon = {"BUY_CALL": "🟢", "SELL_PUT": "🟡",
                "EXIT": "🔴"}.get(sig_type_viz, "⚪")
        st.metric("信号", f"{icon} {sig_disp}")

    # ── OI 因子 (今日预测模式) ──
    oi_factors = None
    oi_adj_upper = oi_adj_lower = None
    oi_adj_bp030 = oi_adj_bp090 = 0
    oi_details = None
    oi_chart_levels = None
    oi_daily = None
    oi_events = []
    oi_hist_bands = None
    eod_df = None
    snap_date = None

    if mode == "今日预测" and pred_u_pct is not None:
        cfg = load_config()
        eod_df, snap_date = load_latest_eod_snapshot(cfg)
        if eod_df is not None:
            oi_factors = compute_oi_factors(eod_df, last_close,
                                            ref_date=today_sgt)
            if oi_factors is not None:
                tu = last_close * (1 + pred_u_pct / 100)
                tl = last_close * (1 + pred_l_pct / 100)
                oi_adj_upper, oi_adj_lower, oi_details = \
                    adjust_range(tu, tl, last_close, oi_factors)

                # 逐日修正 (非平矩形)
                oi_daily, oi_events = adjust_range_daily(
                    tu, tl, last_close, oi_factors, n_days=5)

                # OI 修正后的 band 阈值
                if next_upper > next_lower:
                    adj_nu, adj_nl, oi_band_details = adjust_range(
                        next_upper, next_lower, last_close, oi_factors)
                    if adj_nu > adj_nl:
                        oi_adj_bp030 = adj_nl + 0.30 * (adj_nu - adj_nl)
                        oi_adj_bp090 = adj_nl + 0.90 * (adj_nu - adj_nl)

                oi_chart_levels = {
                    "max_pain": oi_factors["max_pain"],
                    "call_wall": oi_factors["call_wall"],
                    "put_wall": oi_factors["put_wall"],
                }

        # 历史 band OI 修正 (用所有快照)
        all_snaps = load_all_eod_snapshots(cfg)
        if all_snaps:
            adj_ub_hist, adj_lb_hist = adjust_band_history(
                upper_band, lower_band, close, all_snaps)
            if len(adj_ub_hist) > 0:
                oi_hist_bands = (adj_ub_hist, adj_lb_hist)

    # ── 主图表 ──
    # 历史回看长周期用空信号 (不显示交易标注)
    _bc = locals().get("buy_call_viz", buy_call)
    _sp = locals().get("sell_put_viz", sell_put)
    _ex = locals().get("exit_sig_viz", exit_sig)

    fig, trades = generate_chart(
        close, high, viz_dates, upper_band, lower_band,
        _bc, _sp, _ex,
        rv_pctile, regime,
        pred_u_pct=pred_u_pct, pred_l_pct=pred_l_pct,
        show_future=show_future, today=today_for_chart,
        today_close=today_close_chart,
        next_bp030=next_bp030, next_bp090=next_bp090,
        signal_type=sig_type_viz, today_rv=today_rv_chart,
        oi_levels=oi_chart_levels, oi_daily_range=oi_daily,
        oi_events=oi_events, oi_adj_bands=oi_hist_bands,
        oi_adj_bp030=oi_adj_bp030, oi_adj_bp090=oi_adj_bp090)

    st.pyplot(fig, use_container_width=True)

    # ── 导出 ──
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    st.download_button("下载 PNG", buf.getvalue(),
                       file_name="gld_dashboard.png", mime="image/png")
    plt.close(fig)

    # ── 预测模式额外内容 ──
    if mode == "今日预测":
        st.divider()
        c_a, c_b = st.columns(2)

        with c_a:
            st.subheader("5日区间预测")
            if pred_u_pct is not None:
                tu = last_close * (1 + pred_u_pct / 100)
                tl = last_close * (1 + pred_l_pct / 100)
                if oi_adj_upper is not None:
                    adj_u_pct = (oi_adj_upper / last_close - 1) * 100
                    adj_l_pct = (oi_adj_lower / last_close - 1) * 100
                    st.markdown(f"""
| 指标 | 模型预测 | OI修正后 |
|------|---------|---------|
| 预测日期 | {last_date.date()} (基于{today_sgt}) | |
| 上界 | ${tu:.2f} (+{pred_u_pct:.1f}%) | **${oi_adj_upper:.2f}** ({adj_u_pct:+.1f}%) |
| 下界 | ${tl:.2f} ({pred_l_pct:.1f}%) | **${oi_adj_lower:.2f}** ({adj_l_pct:+.1f}%) |
| 区间宽度 | ${tu - tl:.2f} | ${oi_adj_upper - oi_adj_lower:.2f} |
""")
                else:
                    st.markdown(f"""
| 指标 | 值 |
|------|-----|
| 预测日期 | {last_date.date()} (基于{today_sgt}) |
| 上界 | **${tu:.2f}** (+{pred_u_pct:.1f}%) |
| 下界 | **${tl:.2f}** ({pred_l_pct:.1f}%) |
| 区间宽度 | ${tu - tl:.2f} ({pred_u_pct - pred_l_pct:.1f}%) |
""")

                # 逐日区间明细
                if oi_daily is not None and len(oi_daily) > 0:
                    future_dates = pd.bdate_range(
                        pd.Timestamp(today_sgt) + timedelta(days=1),
                        periods=len(oi_daily))
                    daily_rows = []
                    for i, (du, dl) in enumerate(oi_daily):
                        d_pct_u = (du / last_close - 1) * 100
                        d_pct_l = (dl / last_close - 1) * 100
                        event = ""
                        for ei, edesc in oi_events:
                            if ei == i + 1:
                                event = edesc
                        daily_rows.append({
                            "交易日": future_dates[i].strftime("%m/%d"),
                            "上界": f"${du:.1f} ({d_pct_u:+.1f}%)",
                            "下界": f"${dl:.1f} ({d_pct_l:+.1f}%)",
                            "宽度": f"${du - dl:.1f}",
                            "事件": event,
                        })
                    st.markdown("**逐日区间 (OI修正):**")
                    st.dataframe(pd.DataFrame(daily_rows),
                                 use_container_width=True, hide_index=True)

        with c_b:
            st.subheader("下一交易日阈值")
            if next_bp030 > 0:
                zone = "买入区" if last_bp < 0.30 \
                    else ("平仓区" if last_bp > 0.90 else "观望区")
                if oi_adj_bp030 > 0:
                    st.markdown(f"""
| 指标 | 模型原始 | OI修正后 |
|------|---------|---------|
| Band 上界 | ${next_upper:.2f} | |
| Band 下界 | ${next_lower:.2f} | |
| **买入 (bp=0.30)** | ${next_bp030:.2f} | **< ${oi_adj_bp030:.2f}** |
| **平仓 (bp=0.90)** | ${next_bp090:.2f} | **> ${oi_adj_bp090:.2f}** |
| 当前位置 | bp={last_bp:.3f} ({zone}) | |
""")
                else:
                    st.markdown(f"""
| 指标 | 价位 |
|------|------|
| Band 上界 | ${next_upper:.2f} |
| Band 下界 | ${next_lower:.2f} |
| **买入 (bp=0.30)** | **< ${next_bp030:.2f}** |
| **平仓 (bp=0.90)** | **> ${next_bp090:.2f}** |
| 当前位置 | bp={last_bp:.3f} ({zone}) |
""")

        # ── 跨市场价位换算 ──
        if next_bp030 > 0 and gc_gld_ratio is not None:
            st.divider()
            st.subheader("跨市场价位换算")

            # 实时行情 (5分钟缓存)
            rt = _get_realtime_prices()
            if rt is not None:
                _ratio = rt["gc_price"] / last_close
                _cny = rt["usdcny"]
                rt_label = f"实时 ({rt['timestamp']})"
            else:
                _ratio = gc_gld_ratio
                _cny = usdcny_rate if usdcny_rate else 7.0
                rt_label = "近60日均值"
            _g = 31.1035  # troy oz → gram

            def _cvt(gld_price):
                xau = gld_price * _ratio
                gc = xau
                shfe = xau * _cny / _g
                return xau, gc, shfe

            # 统一使用 OI 修正后的阈值 (如有)
            eff_buy = oi_adj_bp030 if oi_adj_bp030 > 0 else next_bp030
            eff_exit = oi_adj_bp090 if oi_adj_bp090 > 0 else next_bp090

            xau_upper, gc_upper, shfe_upper = _cvt(next_upper)
            xau_lower, gc_lower, shfe_lower = _cvt(next_lower)
            xau_buy, gc_buy, shfe_buy = _cvt(eff_buy)
            xau_exit, gc_exit, shfe_exit = _cvt(eff_exit)

            # 当前行 (实时有就用实时)
            if rt is not None:
                xau_now = rt["gc_price"]
                gc_now = rt["gc_price"]
                shfe_now = rt["shfe_approx"]
            else:
                xau_now, gc_now, shfe_now = _cvt(last_close)

            buy_suffix = f" (原${next_bp030:.2f})" \
                if oi_adj_bp030 > 0 else ""
            exit_suffix = f" (原${next_bp090:.2f})" \
                if oi_adj_bp090 > 0 else ""

            st.markdown(f"""
| 价位 | GLD (USD) | 伦敦金现 XAU (USD/oz) | 纽约金 COMEX (USD/oz) | 沪金 AU (CNY/g) |
|------|-----------|----------------------|----------------------|----------------|
| Band 上界 | ${next_upper:.2f} | ${xau_upper:.1f} | ${gc_upper:.1f} | ¥{shfe_upper:.2f} |
| Band 下界 | ${next_lower:.2f} | ${xau_lower:.1f} | ${gc_lower:.1f} | ¥{shfe_lower:.2f} |
| **买入 bp<0.30** | **${eff_buy:.2f}**{buy_suffix} | **${xau_buy:.1f}** | **${gc_buy:.1f}** | **¥{shfe_buy:.2f}** |
| **平仓 bp>0.90** | **${eff_exit:.2f}**{exit_suffix} | **${xau_exit:.1f}** | **${gc_exit:.1f}** | **¥{shfe_exit:.2f}** |
| 当前价 | ${last_close:.2f} | ${xau_now:.1f} | ${gc_now:.1f} | ¥{shfe_now:.2f} |
""")
            src = (f"COMEX GC=${rt['gc_price']:.1f} | "
                   f"USD/CNY={_cny:.4f}"
                   if rt else
                   f"GC/GLD={_ratio:.4f} (近60日均值)")
            st.caption(
                f"换算来源: {src} ({rt_label}) | "
                f"1盎司={_g}克 | "
                f"伦敦金≈COMEX期货"
            )

        # ── OI 因子详情 ──
        if oi_factors is not None:
            st.divider()
            st.subheader("期权 OI 微观结构")
            oi_c1, oi_c2 = st.columns(2)
            with oi_c1:
                dom_dte = oi_factors["dominant_dte"]
                dom_pct = oi_factors["dominant_oi_pct"]
                st.markdown(f"""
| OI 指标 | 值 |
|--------|-----|
| Max Pain | **${oi_factors['max_pain']:.0f}** |
| Call Wall | **${oi_factors['call_wall']:.0f}** (OI: {oi_factors['total_call_oi']:,}) |
| Put Wall | **${oi_factors['put_wall']:.0f}** (OI: {oi_factors['total_put_oi']:,}) |
| PCR | {oi_factors['pcr']:.2f} |
| **主导到期** | **DTE={dom_dte}天** (占OI {dom_pct:.0f}%) |
| 最近到期 | DTE={oi_factors['nearest_dte']}天 |
| Net GEX | {oi_factors['net_gex']:,.0f} |
""")
                # 到期日 OI 分布表
                eb = oi_factors.get("expiry_breakdown", [])
                if eb:
                    st.markdown("**到期日 OI 分布:**")
                    eb_rows = []
                    for e in eb:
                        is_dom = "**" if e["dte"] == dom_dte else ""
                        eb_rows.append({
                            "到期日": e["date"],
                            "DTE": f"{is_dom}{e['dte']}天{is_dom}",
                            "OI": f"{is_dom}{e['oi']:,}{is_dom}",
                            "占比": f"{is_dom}{e['pct']:.1f}%{is_dom}",
                            "类型": e["label"],
                        })
                    st.dataframe(pd.DataFrame(eb_rows),
                                 use_container_width=True,
                                 hide_index=True)
            with oi_c2:
                if oi_details and oi_details.get("adjusted"):
                    adj_lines = "\n".join(
                        f"- {a}" for a in oi_details["adjustments"])
                    st.markdown(f"""**区间修正明细:**

{adj_lines}

| 修正 | 变化 |
|------|------|
| 上界变化 | {oi_details['upper_change_pct']:+.2f}% |
| 下界变化 | {oi_details['lower_change_pct']:+.2f}% |
| 到期因子 | {oi_details['expiry_factor']:.2f} |
""")

            st.caption(
                "OI修正: Max Pain引力 + Call Wall压制 + Put Wall支撑 + "
                "Gamma压缩/放大 | 到期临近效应增强"
            )

            # 关键期权日期说明
            dte = oi_factors["dominant_dte"]
            net_gex = oi_factors["net_gex"]
            mp = oi_factors["max_pain"]
            cw = oi_factors["call_wall"]
            pw = oi_factors["put_wall"]
            pcr = oi_factors["pcr"]

            # 到期日期推算 (基于今日 SGT)
            _today_ts = pd.Timestamp(today_sgt)
            opex_date = pd.bdate_range(
                _today_ts + timedelta(days=1), periods=dte)[-1] \
                if dte > 0 else _today_ts
            # 如果有 expiry_breakdown, 直接用真实日期
            eb = oi_factors.get("expiry_breakdown", [])
            for e in eb:
                if e["dte"] == dte and e["date"]:
                    opex_date = pd.Timestamp(e["date"])
                    break
            opex_str = opex_date.strftime("%m/%d (%a)")

            lines = []

            # ── 概念说明 ──
            lines.append("---")
            lines.append("**关键概念:**")
            lines.append("")

            # Max Pain
            mp_diff = mp - last_close
            mp_pct = mp_diff / last_close * 100
            if mp < last_close:
                mp_effect = (
                    f"Max Pain ${mp:.0f} **低于**现价 ${last_close:.0f} "
                    f"({mp_pct:+.1f}%) → **下拉引力**。"
                    f"做市商净卖出的Call处于实值，"
                    f"需要卖出标的对冲delta，形成卖压。"
                    f"价格越远离Max Pain，对冲卖盘越重，"
                    f"越接近到期引力越强")
            else:
                mp_effect = (
                    f"Max Pain ${mp:.0f} **高于**现价 ${last_close:.0f} "
                    f"({mp_pct:+.1f}%) → **上拉引力**。"
                    f"做市商净卖出的Put处于实值，"
                    f"需要买入标的对冲delta，形成买盘。"
                    f"价格越远离Max Pain，对冲买盘越重，"
                    f"越接近到期引力越强")
            lines.append(
                f"- **Max Pain** (最大痛点): "
                f"令期权买方总亏损最大的结算价。做市商作为"
                f"期权净卖方，其对冲行为会将价格拉向此处。"
                f"**当前: {mp_effect}**")
            lines.append("")

            # Call Wall
            cw_dist = (cw / last_close - 1) * 100
            lines.append(
                f"- **Call Wall** (看涨墙): "
                f"Call OI 最集中的行权价 = ${cw:.0f} "
                f"(距现价 {cw_dist:+.1f}%)。"
                f"做市商在此卖出大量Call，价格接近时需大量"
                f"卖出标的对冲 → 形成上方阻力。"
                f"**{'当前已接近Call Wall，上涨阻力大' if cw_dist < 5 else '距离尚远，压力有限'}**")
            lines.append("")

            # Put Wall
            pw_dist = (1 - pw / last_close) * 100
            lines.append(
                f"- **Put Wall** (看跌墙): "
                f"Put OI 最集中的行权价 = ${pw:.0f} "
                f"(距现价 -{pw_dist:.1f}%)。"
                f"做市商在此卖出大量Put，价格接近时需大量"
                f"买入标的对冲 → 形成下方支撑。"
                f"**{'当前接近Put Wall，下跌有支撑' if pw_dist < 5 else '距离较远，支撑效应弱'}**")
            lines.append("")

            # Gamma
            if net_gex > 0:
                gex_explain = (
                    "做市商持有正Gamma → 价格涨时卖、跌时买 → "
                    "**压缩波动** (区间收窄)")
            else:
                gex_explain = (
                    "做市商持有负Gamma → 价格涨时买、跌时卖 → "
                    "**放大波动** (区间扩大)")
            lines.append(
                f"- **Net GEX** (净Gamma敞口): "
                f"{oi_factors['net_gex']:,.0f}。{gex_explain}")
            lines.append("")

            # ── 压制方向分析 ──
            lines.append("---")
            lines.append("**OI 压制方向分析:**")
            lines.append("")

            # 双向压制: 上有Call Wall, 下有Max Pain/Put Wall
            if mp < last_close:
                lines.append(
                    f"- **下行压力**: Max Pain ${mp:.0f} 在下方"
                    f" → 对冲卖盘将价格拉低。"
                    f"{'同时Put Wall $' + f'{pw:.0f}' + '在更下方提供支撑，限制跌幅' if pw < mp else ''}")
            else:
                lines.append(
                    f"- **下行支撑**: Max Pain ${mp:.0f} 在上方"
                    f" → 对冲买盘托底，下跌空间有限")

            if cw_dist < 8:
                lines.append(
                    f"- **上行阻力**: Call Wall ${cw:.0f} 距现价仅"
                    f" {cw_dist:.1f}% → 做市商卖出标的对冲，压制上涨")
            else:
                lines.append(
                    f"- **上行空间**: Call Wall ${cw:.0f} 距现价"
                    f" {cw_dist:.1f}%，短期阻力不大")

            # 结论: 区间被压缩还是有方向性?
            if mp < last_close and cw_dist < 8:
                lines.append(
                    f"- **结论: 上下夹击** — "
                    f"Max Pain下拉 + Call Wall封顶 → "
                    f"价格大概率在 ${mp:.0f}~${cw:.0f} 区间震荡")
            elif mp > last_close and pw_dist < 5:
                lines.append(
                    f"- **结论: 上下夹击** — "
                    f"Max Pain上拉 + Put Wall托底 → "
                    f"价格大概率在 ${pw:.0f}~${mp:.0f} 区间震荡")
            elif mp < last_close:
                lines.append(
                    f"- **结论: 偏空** — "
                    f"Max Pain在下方施加引力，注意回调风险")
            else:
                lines.append(
                    f"- **结论: 偏多** — "
                    f"Max Pain在上方支撑，有利多头")

            lines.append(f"- PCR = {pcr:.2f}: "
                         + ("看跌情绪偏重" if pcr > 1.2
                            else ("看涨情绪偏重" if pcr < 0.7
                                  else "多空均衡")))

            # ── 到期日与释放时点 ──
            dom_oi = oi_factors["dominant_oi"]
            dom_pct = oi_factors["dominant_oi_pct"]
            near_dte = oi_factors["nearest_dte"]

            lines.append("")
            lines.append("---")
            lines.append("**到期日与压力释放:**")
            lines.append("")
            lines.append(
                f"- **主导到期: {opex_str}** (DTE={dte}天, "
                f"OI={dom_oi:,}, 占比{dom_pct:.0f}%)")

            if near_dte < dte:
                lines.append(
                    f"- 最近到期: DTE={near_dte}天 "
                    f"(OI 较小, 影响有限)")

            # 为什么是这个日期
            lines.append("")
            lines.append(
                f"**为什么是 {opex_str} 而不是其他日期?**")
            if dom_pct > 30:
                lines.append(
                    f"- GLD 期权有**月度**到期 (每月第三个周五) "
                    f"和**周度**到期")
                lines.append(
                    f"- 当前这个到期日集中了 **{dom_pct:.0f}%** 的总 OI "
                    f"({dom_oi:,} 张合约)")
                lines.append(
                    f"- 其他周度到期 OI 通常不到 5%,"
                    f"对冲量太小, 做市商行为不足以影响市场")
                lines.append(
                    f"- OI 越集中 → 做市商对冲量越大 → "
                    f"pin效应越强 → **这个日期最关键**")
            else:
                lines.append(
                    f"- 当前没有特别集中的到期日"
                    f" (最大占比仅{dom_pct:.0f}%), "
                    f"pin效应相对分散")

            lines.append("")
            if dte <= 3:
                lines.append(
                    f"**即将到期** — pin效应最强，"
                    f"价格高度锚定 Max Pain ${mp:.0f}")
                lines.append(
                    f"- 到期当日结算后，{dom_oi:,}张合约 OI "
                    f"全部清算，对冲头寸解除 → 压制**瞬间消失**")
                lines.append(
                    f"- 到期后第1个交易日波动率倾向扩大 "
                    f"(压缩弹簧释放)")
            elif dte <= 7:
                lines.append(
                    f"**{dte}天后到期** — pin效应逐日增强，"
                    f"最后3天最显著")
                lines.append(
                    f"- {opex_str} 结算后, {dom_oi:,}张合约清算,"
                    f" 压制释放")
                lines.append(
                    f"- 新的OI分布 (下一月度到期日) 接管定价权")
            else:
                lines.append(
                    f"距到期还有 {dte} 天，pin效应中等")
                lines.append(
                    f"- 关注 DTE<7 后效应显著增强")

            lines.append(
                f"- 释放后走势取决于: 下一到期日OI分布 + "
                f"基本面方向")

            # ── 5日窗口事件 ──
            if oi_events:
                lines.append("")
                lines.append("**5日预测窗口内事件:**")
                future_bd = pd.bdate_range(
                    pd.Timestamp(today_sgt) + timedelta(days=1), periods=5)
                for di, desc in oi_events:
                    if di <= len(future_bd):
                        lines.append(
                            f"- {future_bd[di-1].strftime('%m/%d (%a)')} "
                            f"(Day {di}): {desc}")

            st.markdown("\n".join(lines))

        # 期权策略
        st.divider()
        st.subheader("期权策略推荐")
        if eod_df is None:
            cfg = load_config()
            eod_df, snap_date = load_latest_eod_snapshot(cfg)

        _render_options_section(eod_df, snap_date, last_close,
                                next_bp090, oi_adj_bp090,
                                gc_gld_ratio, today_sgt, sig_type_viz)

    # ── 近期信号 ──
    st.divider()
    st.subheader("近期信号")
    records = []
    for d in bp_dates[-15:]:
        s = ""
        if buy_call.get(d, False):
            s = "BUY CALL"
        elif sell_put.get(d, False):
            s = "SELL PUT"
        if exit_sig.get(d, False):
            s += " + EXIT" if s else "EXIT"
        records.append({
            "日期": d.strftime("%Y-%m-%d"),
            "GLD": f"${close.get(d, 0):.2f}",
            "bp": f"{bp.get(d, 0):.3f}",
            "Regime": regime.get(d, "?"),
            "RV%": f"{rv_pctile.get(d, 0):.0%}",
            "信号": s if s else "—",
        })
    st.dataframe(pd.DataFrame(records), use_container_width=True,
                 hide_index=True)

    # ── 交易记录 ──
    if trades:
        st.divider()
        st.subheader(f"交易记录 ({len(trades)} 笔)")
        trecs = []
        for t in trades:
            trecs.append({
                "入场": t["entry_date"].strftime("%m/%d"),
                "类型": t["sig_type"],
                "入场价": f"${t['entry_price']:.2f}",
                "出场": t["exit_date"].strftime("%m/%d"),
                "退出": t["exit_type"],
                "出场价": f"${t['exit_price']:.2f}",
                "收益": f"{t['gain']:+.1f}%",
                "持仓": f"{t['hold_days']}d",
            })
        st.dataframe(pd.DataFrame(trecs), use_container_width=True,
                     hide_index=True)


if __name__ == "__main__":
    main()
