"""
金银交易仪表板

Streamlit 交互式界面:
  - 今日预测: 5日区间 + 信号 + 期权策略
  - 历史回看: 自定义时间范围可视化

用法:
    conda activate gold
    streamlit run app.py
"""
import os
import time
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
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

warnings.filterwarnings("ignore")

from core.data import (load_config, load_features, load_gld,
                       load_oos_predictions, load_latest_eod_snapshot,
                       load_all_eod_snapshots, load_gold_futures,
                       load_usdcny, fetch_realtime_gold_fx,
                       auto_refresh_market_data, get_today_sgt,
                       update_features_full, extend_oos_predictions)
from core.regime import RegimeClassifier
from core.signals import build_band, compute_rv_pctile, generate_signals
from core.signals_1h import build_band_1h, generate_signals_1h, backtest_1h
from core.options import get_strategy_table
from core.oi_factors import (compute_oi_factors, adjust_range,
                             adjust_range_daily, adjust_band_history)
from core.training_status import (get_model_age_days, is_stale, is_training,
                                  start_training, stop_training,
                                  get_training_log, get_training_elapsed,
                                  DEFAULT_MAX_AGE_DAYS)

# ── 中文字体 ──
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang HK",
                                "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="金银交易仪表板", page_icon="📊",
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

    # v3.7.62: SI=F / SLV 比率 (跟 GC/GLD 同步管理)
    si_slv_ratio = None
    try:
        import os as _os
        slv_path = _os.path.join(cfg["data_root"], "raw/market/slv.csv")
        si_path = _os.path.join(cfg["data_root"], "raw/market/silver.csv")
        if _os.path.exists(slv_path) and _os.path.exists(si_path):
            slv_df = pd.read_csv(slv_path, index_col=0, parse_dates=True)
            si_df = pd.read_csv(si_path, index_col=0, parse_dates=True)
            si_common = slv_df.index.intersection(si_df.index)
            if len(si_common) > 20:
                si_slv_ratio = float(
                    (si_df.loc[si_common[-60:], "Close"]
                     / slv_df.loc[si_common[-60:], "Close"]).mean())
    except Exception:
        pass

    return (gld, range_df, regime, rv_pctile, gc_gld_ratio, usdcny_rate,
             si_slv_ratio)


@st.cache_data(ttl=300)
def _get_realtime_prices(futures_ticker="GC=F"):
    """实时期货价格+汇率 (5分钟缓存)."""
    return fetch_realtime_gold_fx(futures_ticker)


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
                   oi_adj_bp030=0, oi_adj_bp090=0,
                   asset_key="GLD",
                   spot_ratio=1.0, spot_label=None,
                   straddle_dates=None,
                   unified_markers=None):
    """主图.

    Args:
        spot_ratio: 价位换算比例 (期货/ETF). 1.0 = 不换算.
        spot_label: 现货标签 (如 "伦敦金"). None = 用 asset_key.
        straddle_dates: pd.DatetimeIndex 或 list, Straddle 触发日期 (★ 标记).
    """
    _r = float(spot_ratio) if spot_ratio else 1.0
    _disp_label = spot_label if spot_label else asset_key
    _unit = "USD/oz" if spot_label else "USD"
    _label = f"{_disp_label} ({_unit})"
    _title_prefix = _disp_label
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

    # 价格 (×_r)
    cl = close.reindex(dates_all).dropna()
    ax.plot(xi_arr(cl.index), cl.values * _r, "k-",
            lw=1.8, alpha=0.9, zorder=3)

    # Band (原始, ×_r)
    ub = upper_band.reindex(dates_all).dropna()
    lb = lower_band.reindex(dates_all).dropna()
    ax.plot(xi_arr(ub.index), ub.values * _r,
            color="green", lw=1, alpha=0.5)
    ax.plot(xi_arr(lb.index), lb.values * _r,
            color="magenta", lw=1, alpha=0.5)
    cidx = ub.index.intersection(lb.index)
    if len(cidx) > 0:
        ax.fill_between(xi_arr(cidx),
                         lb.loc[cidx].values * _r,
                         ub.loc[cidx].values * _r,
                         alpha=0.06, color="green")

    # Band (OI 修正, ×_r)
    if oi_adj_bands is not None:
        adj_ub, adj_lb = oi_adj_bands
        adj_ub = adj_ub.reindex(dates_all).dropna()
        adj_lb = adj_lb.reindex(dates_all).dropna()
        if len(adj_ub) > 0:
            ax.plot(xi_arr(adj_ub.index), adj_ub.values * _r,
                    color="darkgreen", lw=1.5, ls="--", alpha=0.7)
            ax.plot(xi_arr(adj_lb.index), adj_lb.values * _r,
                    color="darkmagenta", lw=1.5, ls="--", alpha=0.7)
            aidx = adj_ub.index.intersection(adj_lb.index)
            if len(aidx) > 0:
                ax.fill_between(xi_arr(aidx),
                                 adj_lb.loc[aidx].values * _r,
                                 adj_ub.loc[aidx].values * _r,
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

    # Exit 信号 (×_r)
    entry_dates_set = set(t["entry_date"] for t in trades)
    ex_dates = [d for d in dates_all if exit_sig.get(d, False)
                and d not in entry_dates_set]
    if ex_dates:
        ax.scatter(xi_arr(ex_dates),
                   [cl.get(d, np.nan) * _r for d in ex_dates],
                   marker="v", s=120, color="#F44336", edgecolors="darkred",
                   linewidths=0.7, zorder=5)
        for d in ex_dates:
            v = cl.get(d, np.nan)
            if not np.isnan(v) and xi(d) is not None:
                ax.annotate(d.strftime("%m/%d"), xy=(xi(d), v * _r),
                            xytext=(0, 12), textcoords="offset points",
                            fontsize=6, ha="center", color="#F44336",
                            fontweight="bold")

    # 交易轨迹 (×_r)
    for t in trades:
        td = [xi(x[0]) for x in t["trajectory"] if xi(x[0]) is not None]
        tp = [x[1] * _r for x in t["trajectory"] if xi(x[0]) is not None]
        c = SIG_COLORS[t["sig_type"]]
        ax.plot(td, tp, "-", color=c, lw=2,
                alpha=0.85 if t["gain"] > 0 else 0.4, zorder=4)
        ei = xi(t["entry_date"])
        if ei is not None:
            ax.scatter([ei], [t["entry_price"] * _r], marker="^",
                       s=160, color=c, edgecolors="black", linewidths=0.7,
                       zorder=6)
            rv_val = rv_pctile.get(t["entry_date"], np.nan)
            rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
            ax.annotate(
                f"{t['entry_date'].strftime('%m/%d')}{rv_txt}",
                xy=(ei, t["entry_price"] * _r),
                xytext=(0, -16), textcoords="offset points",
                fontsize=7, ha="center", color=c, fontweight="bold")
        exi = xi(t["exit_date"])
        if exi is not None:
            mk, mc = EXIT_MARKERS.get(t["exit_type"], ("o", "gray"))
            if t["exit_date"] not in entry_dates_set:
                ax.scatter([exi], [t["exit_price"] * _r], marker=mk,
                           s=100, color=mc, edgecolors="black",
                           linewidths=0.5, zorder=7)
            oy = 16 if t["exit_date"] in entry_dates_set \
                else (12 if t["gain"] > 0 else -14)
            ax.annotate(f"{t['gain']:+.1f}% ({t['hold_days']}d)",
                        xy=(exi, t["exit_price"] * _r),
                        xytext=(5, oy), textcoords="offset points",
                        fontsize=7, color=c, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.8, ec="none"))

    # ── Straddle 标记 (★) — 仅当 straddle_dates 提供时绘制 ──
    if straddle_dates is not None and len(straddle_dates) > 0:
        _str_idx = pd.DatetimeIndex(straddle_dates)
        _plotted = []
        for sd in _str_idx:
            xv = xi(sd)
            if xv is None:
                continue
            yv = cl.get(sd, np.nan)
            if pd.isna(yv):
                continue
            _plotted.append((xv, yv * _r, sd))
        if _plotted:
            xs = [p[0] for p in _plotted]
            ys = [p[1] for p in _plotted]
            ax.scatter(xs, ys, marker="*", s=220, color="#FFD700",
                       edgecolors="black", lw=0.7, zorder=7)
            for xv, yv, sd in _plotted:
                ax.annotate("STRADDLE", xy=(xv, yv),
                            xytext=(0, 14), textcoords="offset points",
                            fontsize=6.5, ha="center", color="#B8860B",
                            fontweight="bold")

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

    # 5日预测区间 (×_r)
    if show_future and today is not None and pred_u_pct is not None:
        tu = today_close * (1 + pred_u_pct / 100)
        tl = today_close * (1 + pred_l_pct / 100)
        fut_xi = xi_arr(future_bdays)

        if oi_daily_range is not None and len(oi_daily_range) > 0 \
                and len(fut_xi) == len(oi_daily_range):
            uppers = [d[0] * _r for d in oi_daily_range]
            lowers = [d[1] * _r for d in oi_daily_range]

            ax.fill_between(fut_xi, lowers, uppers,
                             alpha=0.12, color="gold", zorder=1)
            ax.plot(fut_xi, uppers, color="goldenrod", lw=1.2,
                    ls="--", alpha=0.7)
            ax.plot(fut_xi, lowers, color="goldenrod", lw=1.2,
                    ls="--", alpha=0.7)

            # 原始模型区间 (灰色虚线参考)
            ax.plot([fut_xi[0], fut_xi[-1]], [tu * _r, tu * _r],
                    color="gray", lw=0.8, ls=":", alpha=0.4)
            ax.plot([fut_xi[0], fut_xi[-1]], [tl * _r, tl * _r],
                    color="gray", lw=0.8, ls=":", alpha=0.4)

            # 标注最终日
            ax.annotate(
                f"${uppers[-1]:.0f} "
                f"({(uppers[-1]/(today_close*_r)-1)*100:+.1f}%)",
                xy=(fut_xi[-1], uppers[-1]), fontsize=8, fontweight="bold",
                color="goldenrod", ha="right", va="bottom")
            ax.annotate(
                f"${lowers[-1]:.0f} "
                f"({(lowers[-1]/(today_close*_r)-1)*100:+.1f}%)",
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
            # 无 OI: 平矩形 (×_r)
            if len(fut_xi) >= 2:
                tu_r, tl_r = tu * _r, tl * _r
                ax.fill_between([fut_xi[0], fut_xi[-1]],
                                 [tl_r, tl_r], [tu_r, tu_r],
                                 alpha=0.12, color="gold", zorder=1)
                ax.plot([fut_xi[0], fut_xi[-1]], [tu_r, tu_r],
                        color="goldenrod", lw=1.2, ls="--", alpha=0.7)
                ax.plot([fut_xi[0], fut_xi[-1]], [tl_r, tl_r],
                        color="goldenrod", lw=1.2, ls="--", alpha=0.7)
                ax.annotate(f"${tu_r:.0f} (+{pred_u_pct:.1f}%)",
                            xy=(fut_xi[-1], tu_r), fontsize=8,
                            fontweight="bold", color="goldenrod",
                            ha="right", va="bottom")
                ax.annotate(f"${tl_r:.0f} ({pred_l_pct:.1f}%)",
                            xy=(fut_xi[-1], tl_r), fontsize=8,
                            fontweight="bold", color="goldenrod",
                            ha="right", va="top")

    # 下一交易日阈值线 (×_r)
    last_xi = len(dates_all) - 1
    # 使用 OI 修正后的阈值 (如有), 否则用原始值
    eff_bp030 = oi_adj_bp030 if oi_adj_bp030 > 0 else next_bp030
    eff_bp090 = oi_adj_bp090 if oi_adj_bp090 > 0 else next_bp090
    if eff_bp030 > 0:
        eff_bp030_d = eff_bp030 * _r
        ax.axhline(eff_bp030_d, color="#2196F3", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        buy_label = f"BUY < ${eff_bp030_d:.1f}"
        if oi_adj_bp030 > 0 and next_bp030 > 0:
            buy_label += f" (原${next_bp030 * _r:.1f})"
        ax.annotate(buy_label,
                    xy=(last_xi, eff_bp030_d),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#2196F3", fontweight="bold",
                    ha="left", va="center")
    if eff_bp090 > 0:
        eff_bp090_d = eff_bp090 * _r
        ax.axhline(eff_bp090_d, color="#F44336", lw=1.2, ls="-.",
                   alpha=0.5, zorder=2)
        exit_label = f"EXIT > ${eff_bp090_d:.1f}"
        if oi_adj_bp090 > 0 and next_bp090 > 0:
            exit_label += f" (原${next_bp090 * _r:.1f})"
        ax.annotate(exit_label,
                    xy=(last_xi, eff_bp090_d),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color="#F44336", fontweight="bold",
                    ha="left", va="center")

    # 当日标注 (×_r)
    if today is not None and today_close is not None and xi(today) is not None:
        ub_v = upper_band.get(today, 0)
        lb_v = lower_band.get(today, 0)
        bp_v = (today_close - lb_v) / (ub_v - lb_v) \
            if ub_v != lb_v else 0

        mc = {"BUY_CALL": "#2196F3", "SELL_PUT": "#FF9800",
              "EXIT": "#F44336"}.get(signal_type, "black")
        today_close_d = today_close * _r
        ax.scatter([xi(today)], [today_close_d], marker="D", s=120,
                   color=mc, edgecolors="black", linewidths=1.5, zorder=8)
        sl = {"BUY_CALL": "BUY CALL", "SELL_PUT": "SELL PUT",
              "EXIT": "EXIT"}.get(signal_type, "")
        ax.annotate(
            f"${today_close_d:.1f}  bp={bp_v:.2f}  RV={today_rv:.0%}"
            + (f"\n{sl}" if sl else ""),
            xy=(xi(today), today_close_d), xytext=(-60, -28),
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
            mp_d = mp * _r
            ax.axhline(mp_d, color="orange", lw=1, ls=":", alpha=0.6, zorder=2)
            ax.annotate(f"Max Pain ${mp_d:.0f}", xy=(last_xi, mp_d),
                        xytext=(10, 6), textcoords="offset points",
                        fontsize=7, color="orange", fontweight="bold")
        if cw:
            cw_d = cw * _r
            ax.axhline(cw_d, color="red", lw=1, ls=":", alpha=0.5, zorder=2)
            ax.annotate(f"Call Wall ${cw_d:.0f}", xy=(last_xi, cw_d),
                        xytext=(10, 6), textcoords="offset points",
                        fontsize=7, color="red", fontweight="bold")
        if pw:
            pw_d = pw * _r
            ax.axhline(pw_d, color="green", lw=1, ls=":", alpha=0.5, zorder=2)
            ax.annotate(f"Put Wall ${pw_d:.0f}", xy=(last_xi, pw_d),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=7, color="green", fontweight="bold")

    # 格式
    parts = [f"{_title_prefix} 交易仪表板"]
    if today is not None:
        parts.append(today.strftime("%Y-%m-%d"))
        parts.append(f"Regime: {regime.get(today, '?')}")
        if signal_type:
            parts.append(f"信号: {signal_type.replace('_', ' ')}")
    ax.set_title("  |  ".join(parts), fontsize=13, fontweight="bold")
    ax.set_ylabel(_label, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_xlim(-0.5, len(plot_dates) - 0.5)

    legend_el = [
        Line2D([0], [0], color="black", lw=1.5, label=_title_prefix),
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
    if straddle_dates is not None and len(straddle_dates) > 0:
        legend_el.append(Line2D([0], [0], marker="*", color="#FFD700",
                                markersize=12, lw=0, label="★ Straddle (做多 vol)"))
    # v3.7.70: 加 SHORT_VOL legend (✚ 十字)
    if unified_markers is not None and len(unified_markers) > 0:
        legend_el.append(Line2D([0], [0], marker="P", color="#FF6F00",
                                markersize=10, lw=0,
                                label="✚ SHORT_VOL (做空 vol IC)"))
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

    # v3.7.60: 加 dedupe 后 unified markers (含加仓 is_add) — 跟盘中信号 chart 一致
    # 解决: 4-29 谷底 build_trades 因 in_trade 不画新 entry, 此处补全
    # v3.7.70: 混合信号不合并 — 每种 part 画独立 marker
    if unified_markers is not None and len(unified_markers) > 0:
        _SIG_PAL = {"BUY CALL": ("#2196F3", "^"),
                     "SELL PUT": ("#FF9800", "^"),
                     "EXIT": ("#F44336", "v"),
                     "STRADDLE": ("#FFD700", "*"),
                     "SHORT_VOL": ("#FF6F00", "P")}
        for d, row in unified_markers.iterrows():
            ch = row.get("chosen", "")
            if not ch:
                continue
            entry_p = row.get("entry_p", 0)
            is_add = row.get("is_add", False)
            xi_d = xi(d)
            if xi_d is None or entry_p <= 0:
                continue
            # 拆分混合信号 ("BUY CALL + STRADDLE" → ["BUY CALL", "STRADDLE"])
            parts = [p.strip() for p in ch.split("+") if p.strip()]
            # v3.7.70: chosen 可能因 vega 矛盾选了 dir 单独, 但 straddle/short_vol
            # 也触发了 — 额外加 marker (e.g. 4-27 chosen=SELL PUT 但 STRADDLE 也 True)
            if row.get("straddle_signal", False) and "STRADDLE" not in parts:
                parts.append("STRADDLE")
            if row.get("short_vol_signal", False) and "SHORT_VOL" not in parts:
                parts.append("SHORT_VOL")
            for j, part in enumerate(parts):
                color, marker = _SIG_PAL.get(part, ("gray", "o"))
                edge = "purple" if is_add and "STRADDLE" not in part \
                       and "SHORT_VOL" not in part else "black"
                size = (200 if "STRADDLE" in part or "SHORT_VOL" in part
                        else (160 if is_add else 120))
                # 多 part 时 y 偏移避免重叠
                y_offset = j * (entry_p * 0.003)
                ax.scatter([xi_d], [(entry_p + y_offset) * spot_ratio],
                           marker=marker, s=size, color=color,
                           edgecolors=edge,
                           lw=1.5 if is_add else 0.8, zorder=7)
            if is_add and "BUY CALL" in ch.split("+")[0] \
                    or (is_add and "SELL PUT" in ch.split("+")[0]):
                ax.annotate("加仓", xy=(xi_d, entry_p * spot_ratio),
                            xytext=(0, -15), textcoords="offset points",
                            fontsize=7, ha="center", color="purple",
                            fontweight="bold")

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
        if t.get("active") or t.get("exit_date") is None:
            continue
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
                            regime, rv_pctile, gld_1h=None,
                            asset_key="GLD",
                            entry_log_lookup=None,
                            exit_log_lookup=None,
                            primary_chart_mode="log_worst"):
    """真实策略回测 (与 Dashboard 显示策略 1:1 一致), 近6月/1年/2年.

    入场: 盘中 log 真实触发 → bp_low < 0.30 + Stoch RSI/MACD/KDJ 确认.
    退出: bp_high > 0.90 (BandExit) → log EXIT 代表价 / 兜底 bp090 阈值.
    风控: 单笔 -3% 止损 + 连续 2 笔熔断.

    entry_log_lookup / exit_log_lookup: 见 run_backtest 文档.
    主图按 primary_chart_mode 画一条净值线 (默认 log_worst).
    summary 同时跑 4 档对比: log_worst / log_best / log_first / close.
    """
    from core.signals_v2 import run_backtest as run_v22

    _asset_label = "GLD (黄金)" if asset_key == "GLD" else "SLV (白银)"
    last = bp_dates[-1]
    periods = [
        ("近6月", last - pd.DateOffset(months=6)),
        ("近1年", last - pd.DateOffset(years=1)),
        ("近2年", last - pd.DateOffset(years=2)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)
    fig.suptitle(f"{_asset_label} 策略回测 "
                 f"(真实策略: 盘中触发入场 + StopLoss/BandExit/Pullback)",
                 fontsize=15, fontweight="bold")

    summary_rows = []
    summary_by_mode = {}  # {mode_name: [(period_label, period_row_dict), ...]}

    # 4 档入场口径 (跑 4 次)
    # 加载 log 计算 best / first 两档 lookup
    from core.intraday_triggers import (
        load_log as _il_load,
        worst_of_day as _il_worst,
        best_of_day as _il_best,
        first_of_day as _il_first,
    )
    from core.data import load_config as _il_cfg
    _bt_log_path = os.path.join(_il_cfg()["data_root"],
                                 "intraday_signal_log.parquet")
    _bt_log_full = _il_load(_bt_log_path)
    _bt_log_a = _bt_log_full[_bt_log_full["asset"] == asset_key] \
        if len(_bt_log_full) else _bt_log_full
    _lk_worst = entry_log_lookup if entry_log_lookup is not None \
        else (_il_worst(_bt_log_a, "BUY") if len(_bt_log_a) else pd.DataFrame())
    _lk_best = _il_best(_bt_log_a, "BUY") if len(_bt_log_a) else pd.DataFrame()
    _lk_first = _il_first(_bt_log_a, "BUY") if len(_bt_log_a) else pd.DataFrame()

    # (mode_name, entry_price_mode, entry_log_lookup, 中文标签)
    # 注: 旧的 "low_threshold" (min(bp030, lo)) 假设能买在阈值或日内最低,
    # 这种"完美抓底"现实不可能, 已移除. log_best 是真正可达的最优.
    _modes = [
        ("log_worst", "log", _lk_worst, "log 最差 (保守)"),
        ("log_best",  "log", _lk_best,  "log 最优 (可达)"),
        ("log_first", "log", _lk_first, "log 第一次"),
        ("close",     "close", None,    "收盘价"),
    ]

    for ax, (label, start) in zip(axes, periods):
        period_dates = close.index[close.index >= start]
        if len(period_dates) == 0:
            ax.set_title(f"{label}: 无数据")
            continue

        buy_hold = pd.Series(index=period_dates, dtype=float)
        base = close.get(period_dates[0], close.iloc[0])
        for d in period_dates:
            buy_hold[d] = close.get(d, base) / base * 100

        # 5 档全部跑 (后面 summary 用)
        per_mode_stats = {}
        per_mode_trades = {}
        for m_name, m_mode, m_lookup, _label_zh in _modes:
            tr = run_v22(close, high, low, upper_band, lower_band,
                         regime, rv_pctile, gld_1h=gld_1h,
                         start_date=start,
                         entry_log_lookup=m_lookup,
                         exit_log_lookup=exit_log_lookup,
                         entry_price_mode=m_mode)
            per_mode_trades[m_name] = tr
            st_now = _trade_stats(tr, buy_hold)
            per_mode_stats[m_name] = (st_now, tr)

        # 主图用 primary_chart_mode 这条线
        stats, trades = per_mode_stats.get(
            primary_chart_mode, per_mode_stats["log_worst"])

        nav = _build_nav(trades, period_dates)

        # 画图: 买入持有 + 4 档净值线
        ax.plot(buy_hold.index, buy_hold.values, color="gray", lw=1.2,
                alpha=0.5, label="买入持有")
        _line_colors = {
            "log_worst": "#D32F2F",  # 红 (保守)
            "log_best":  "#388E3C",  # 绿 (最优可达)
            "log_first": "#1976D2",  # 蓝 (第一次)
            "close":     "#FF9800",  # 橙 (收盘)
        }
        for m_name, _, _, m_label_zh in _modes:
            m_stats, m_trades = per_mode_stats[m_name]
            if not m_stats:
                continue
            m_nav = _build_nav(m_trades, period_dates)
            ax.plot(m_nav.index, m_nav.values,
                    color=_line_colors[m_name],
                    lw=2 if m_name == primary_chart_mode else 1.2,
                    alpha=1.0 if m_name == primary_chart_mode else 0.55,
                    label=f"{m_label_zh} ({m_stats['n']}笔 "
                          f"{m_stats['total']:+.1f}%)")

        # 交易标注
        for t in trades:
            if t.get("active") or t.get("exit_date") is None:
                continue
            if t["entry_date"] in nav.index:
                ax.scatter([t["entry_date"]], [nav[t["entry_date"]]],
                           marker="^", s=50, color="#FF9800",
                           edgecolors="black", linewidths=0.4, zorder=5)

        # Regime 背景
        reg = regime.reindex(period_dates)
        bull = reg == "Bull"
        if bull.any():
            starts_b = period_dates[bull & (~bull.shift(1, fill_value=False))]
            ends_b = period_dates[bull & (~bull.shift(-1, fill_value=False))]
            for s, e in zip(starts_b, ends_b):
                ax.axvspan(s, e, alpha=0.03, color="green")

        ax.axhline(100, color="black", lw=0.5, ls=":", alpha=0.3)
        ax.set_ylabel("净值 (起始=100)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # 统计文本
        if stats:
            stats_text = (
                f"策略: {stats['total']:+.1f}% ({stats['n']}笔 {stats['wr']:.0%}WR "
                f"均{stats['avg']:+.1f}% 回撤{stats['max_dd']:.1f}% "
                f"持仓{stats['hold']:.1f}d)"
                f"  |  持有: {stats.get('bh', 0):+.1f}%")
        else:
            stats_text = "无交易"

        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(fc="lightyellow", ec="gray", alpha=0.9))
        ax.set_title(f"{label} ({start.strftime('%Y-%m')} ~ "
                     f"{last.strftime('%Y-%m')})", fontsize=12,
                     fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)

        bh_running_max = buy_hold.cummax()
        bh_max_dd = ((buy_hold - bh_running_max) / bh_running_max * 100).min()

        # summary: 按口径分组, 每口径一张独立表 (周期 × 列指标)
        for m_name, _, _, m_label_zh in _modes:
            m_stats, m_trades = per_mode_stats[m_name]
            row = {
                "周期": label,
                "买入持有": f"{m_stats.get('bh', 0):+.1f}%" if m_stats else "—",
                "持有回撤": f"{bh_max_dd:.1f}%",
            }
            if m_stats:
                row["策略收益"] = f"{m_stats['total']:+.1f}%"
                row["交易笔数"] = m_stats["n"]
                row["胜率"] = f"{m_stats['wr']:.0%}"
                row["最大回撤"] = f"{m_stats['max_dd']:.1f}%"
                row["平均/笔"] = f"{m_stats['avg']:+.1f}%"
                row["持仓"] = f"{m_stats['hold']:.1f}d"
                # 来源覆盖: 盘中 log 命中数
                n_intra = sum(1 for t in m_trades
                              if t.get("entry_source", "").startswith("盘中"))
                row["盘中命中"] = f"{n_intra}/{m_stats['n']}"
            else:
                row["策略收益"] = "—"
            summary_rows.append({**row, "入场口径": m_label_zh})
            summary_by_mode.setdefault(m_name, []).append((m_label_zh, row))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, summary_rows, summary_by_mode


def compute_daily_stoch_rsi(close: pd.Series,
                             rsi_period: int = 14,
                             stoch_period: int = 14,
                             smooth_k: int = 3,
                             smooth_d: int = 3):
    """日线 Stoch RSI(14, 14, 3, 3). 返回 (K, D) 两条 Series."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_low = rsi.rolling(stoch_period, min_periods=3).min()
    rsi_high = rsi.rolling(stoch_period, min_periods=3).max()
    k_raw = ((rsi - rsi_low)
             / (rsi_high - rsi_low).replace(0, np.nan)) * 100
    k = k_raw.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return k, d


def render_daily_stoch_rsi(close: pd.Series, asset_key: str,
                            viz_dates=None, lookback: int = 120):
    """日线 Stoch RSI 单子图 (无价格图, 范围与主图一致).

    viz_dates: 主图使用的日期范围 (优先), 否则用 lookback.
    """
    k, d = compute_daily_stoch_rsi(close)
    if viz_dates is not None and len(viz_dates) > 0:
        window_idx = pd.DatetimeIndex(viz_dates)
    else:
        window_idx = close.tail(lookback).index
    window_k = k.reindex(window_idx)
    window_d = d.reindex(window_idx)

    last_k = window_k.dropna().iloc[-1] if window_k.dropna().size else None
    last_d = window_d.dropna().iloc[-1] if window_d.dropna().size else None

    if last_k is None:
        return

    if last_k >= 80:
        zone, color = "超买 (止盈/减仓窗口)", "#E53935"
    elif last_k <= 20:
        zone, color = "超卖 (潜在入场窗口)", "#43A047"
    elif last_k > last_d and last_k < 50:
        zone, color = "超卖区反转 (多头信号)", "#1E88E5"
    elif last_k < last_d and last_k > 50:
        zone, color = "超买区回落 (空头信号)", "#FB8C00"
    else:
        zone, color = "中性", "#757575"

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        st.metric(f"Stoch RSI K", f"{last_k:.0f}",
                  delta=f"{last_k - last_d:+.1f} vs D")
    with col_b:
        st.metric("Stoch RSI D", f"{last_d:.0f}")
    with col_c:
        st.markdown(f"<div style='padding-top:18px'><b>状态: "
                    f"<span style='color:{color}'>{zone}</span></b></div>",
                    unsafe_allow_html=True)

    # 用整数 x 索引, 与主图 (xi_arr) 完全对齐
    _xi_d = np.arange(len(window_idx))
    _idx_list = list(window_idx)

    def _fmt_d(i, pos):
        ii = int(round(i))
        if 0 <= ii < len(_idx_list):
            return _idx_list[ii].strftime("%m/%d")
        return ""

    fig, ax_st = plt.subplots(figsize=(18, 2.5))
    ax_st.axhspan(80, 100, color="#E53935", alpha=0.12)
    ax_st.axhspan(0, 20, color="#43A047", alpha=0.12)
    ax_st.axhline(80, color="#E53935", lw=0.8, ls="--", alpha=0.6)
    ax_st.axhline(20, color="#43A047", lw=0.8, ls="--", alpha=0.6)
    ax_st.axhline(50, color="gray", lw=0.6, ls=":", alpha=0.5)

    ax_st.plot(_xi_d, window_k.values, color="#1E88E5", lw=1.3, label="K")
    ax_st.plot(_xi_d, window_d.values, color="#FB8C00",
               lw=1.1, ls="--", label="D")
    ax_st.scatter([_xi_d[-1]], [last_k], s=40, color=color, zorder=5)

    ax_st.set_ylim(-2, 102)
    ax_st.set_xlim(0, len(_idx_list) - 1)
    ax_st.set_ylabel("Stoch RSI")
    ax_st.grid(True, alpha=0.3)
    ax_st.legend(loc="upper left", fontsize=9)
    ax_st.xaxis.set_major_formatter(FuncFormatter(_fmt_d))
    ax_st.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    plt.setp(ax_st.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
                          gc_gld_ratio, usdcny_rate, today_sgt,
                          asset_key="GLD"):
    """真实策略: Band 开窗 + 盘中触发入场 + bp090 退出 + 3% 止损."""
    from core.signals_v2 import (generate_daily_signals, run_backtest,
                                  PULLBACK_GAIN, PULLBACK_DD,
                                  MAX_HOLD_DAYS)

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

    # 1h 数据 (v3.7.26: csv 过期 > 7 天自动用 yfinance 补齐)
    _1h_fname = "gld_1h.csv" if asset_key == "GLD" else "slv_1h.csv"
    gld_1h_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "Gold", "data", "raw", "market", _1h_fname)
    gld_1h_path = os.path.normpath(gld_1h_path)
    gld_1h = pd.read_csv(gld_1h_path, index_col=0, parse_dates=True) \
        if os.path.exists(gld_1h_path) else None

    # 数据陈旧检查 + yfinance 兜底
    @st.cache_data(ttl=600)
    def _refresh_1h_yfinance(ticker, period="60d", interval="1h"):
        """v3.7.79: 加 interval 参数. 支持 1m/5m/15m/30m/1h.
        yfinance 限制: 1m=7d, 5m/15m/30m=60d, 1h=730d.
        """
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
        return None

    @st.cache_data(ttl=300)
    def _fetch_intraday(ticker, interval, period_days):
        """通用 intraday 数据拉取, 5 min 缓存.
        Args:
            ticker: GLD / SLV
            interval: '1m','5m','15m','30m','1h'
            period_days: 回看天数 (受 yfinance 限制)
        """
        # yfinance period 字符串
        max_p = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}.get(interval, 60)
        p_days = min(period_days, max_p)
        return _refresh_1h_yfinance(ticker, period=f"{p_days}d", interval=interval)

    _today_check = pd.Timestamp.now().normalize()
    _stale_threshold = timedelta(days=7)
    _is_stale = (gld_1h is None or len(gld_1h) == 0
                  or (_today_check - gld_1h.index[-1]) > _stale_threshold)
    if _is_stale:
        _yf_ticker_for_intraday = "GLD" if asset_key == "GLD" else "SLV"
        _yf_1h = _refresh_1h_yfinance(_yf_ticker_for_intraday, period="60d")
        if _yf_1h is not None and len(_yf_1h) > 0:
            if gld_1h is None or len(gld_1h) == 0:
                gld_1h = _yf_1h
            else:
                # Append 不重复
                _last_csv = gld_1h.index[-1]
                _yf_new = _yf_1h[_yf_1h.index > _last_csv]
                if len(_yf_new) > 0:
                    gld_1h = pd.concat([gld_1h, _yf_new])
            st.sidebar.caption(f"⚠️ {_1h_fname} 过期 → yfinance 补齐 "
                               f"({_yf_1h.index[-1].strftime('%m/%d %H:%M')})")

    # 信号 (v3.7.19 实时化: 用 1h 数据 + 实时金价 更新今日 H/L 后再算信号)
    # 让 sig_df 反映 latest 1h close + intraday range, 而非陈旧 daily close
    _close_for_sig = close_d.copy()
    _high_for_sig = high_d.copy()
    _low_for_sig = low_d.copy()

    if gld_1h is not None and len(gld_1h) > 0:
        # 取今日 (last_date 之后) 的 1h 数据, 合成"动态今日 H/L"
        _bp_dates_check = upper_band.dropna().index.intersection(
            lower_band.dropna().index)
        _last_d_bd = _bp_dates_check[-1] if len(_bp_dates_check) else close_d.index[-1]
        _intraday_1h = gld_1h[gld_1h.index > _last_d_bd]
        if len(_intraday_1h) > 0:
            _new_d = _intraday_1h.index[-1].normalize()
            # v3.7.59: 过滤 yfinance prepost 脏数据 (Low 偏离 Close > 5% 是异常)
            _clean = _intraday_1h.copy()
            _clean = _clean[(_clean["Low"] >= _clean["Close"] * 0.95) &
                              (_clean["High"] <= _clean["Close"] * 1.05)]
            if len(_clean) == 0:
                _clean = _intraday_1h
            _new_h = _clean["High"].max()
            _new_l = _clean["Low"].min()
            _new_c = _clean["Close"].iloc[-1]
            # append 一个新的"今日"条目 (实时数据)
            _close_for_sig.loc[_new_d] = _new_c
            _high_for_sig.loc[_new_d] = _new_h
            _low_for_sig.loc[_new_d] = _new_l
            _close_for_sig = _close_for_sig.sort_index()
            _high_for_sig = _high_for_sig.sort_index()
            _low_for_sig = _low_for_sig.sort_index()

    sig_df = generate_daily_signals(
        _close_for_sig, _high_for_sig, _low_for_sig,
        upper_band, lower_band, regime, rv_pctile, asset=asset_key)

    # ── 加载盘中触发 log (在回测之前!), 构造每日代表价 ──
    from core.data import load_config, load_oos_predictions
    _intra_cfg = load_config()
    # v3.7.68/73: 用 dedupe-平均价 + 平均时间 (= 实战分批加仓的均时均价)
    from core.intraday_triggers import (
        load_log as _ig_load,
        average_of_day as _ig_avg_global,
        dedupe_intraday as _ig_dedupe_g)
    _intra_log_path = os.path.join(_intra_cfg["data_root"],
                                    "intraday_signal_log.parquet")
    _intra_log_full = _ig_load(_intra_log_path)
    _intra_log_asset = _intra_log_full[_intra_log_full["asset"] == asset_key] \
        if len(_intra_log_full) else _intra_log_full
    _worst_buy_lookup = _ig_avg_global(_intra_log_asset, "BUY",
                                          dedup_first=True, min_drop_pct=0.3) \
        if len(_intra_log_asset) else pd.DataFrame()
    _worst_exit_lookup = _ig_avg_global(_intra_log_asset, "EXIT",
                                           dedup_first=True, min_drop_pct=0.3) \
        if len(_intra_log_asset) else pd.DataFrame()
    # v3.7.73: 计算每日 BUY/EXIT trigger 平均时间 (用于主图 marker x 位置)
    # v3.7.85: timeframe filter 在 user 选完 interval 后再算 (见 _avg_buy_time =).
    # v3.7.88: 优先 active tf; 该 tf 该日无数据 → fallback 任意 tf
    def _avg_trigger_time(side, tf_match):
        if not len(_intra_log_asset): return {}
        active = _intra_log_asset[
            (_intra_log_asset["side"] == side) &
            (_intra_log_asset.get("timeframe", "") == tf_match)]
        any_tf = _intra_log_asset[_intra_log_asset["side"] == side]
        out = {}
        for src in (active, any_tf):
            for d, grp in src.groupby("date"):
                if pd.Timestamp(d) in out: continue
                dd = _ig_dedupe_g(grp, side=side, min_drop_pct=0.3)
                if len(dd) == 0: continue
                ts_list = pd.to_datetime(dd["trigger_time"])
                out[pd.Timestamp(d)] = pd.Timestamp(int(ts_list.astype("int64").mean()))
        return out
    def _avg_trigger_price(side, tf_match):
        if not len(_intra_log_asset): return {}
        active = _intra_log_asset[
            (_intra_log_asset["side"] == side) &
            (_intra_log_asset.get("timeframe", "") == tf_match)]
        any_tf = _intra_log_asset[_intra_log_asset["side"] == side]
        out = {}
        for src in (active, any_tf):
            for d, grp in src.groupby("date"):
                if pd.Timestamp(d) in out: continue
                dd = _ig_dedupe_g(grp, side=side, min_drop_pct=0.3)
                if len(dd) == 0: continue
                out[pd.Timestamp(d)] = float(pd.to_numeric(dd["price"], errors="coerce").mean())
        return out
    _avg_buy_time: dict = {}; _avg_buy_price: dict = {}
    _avg_exit_time: dict = {}; _avg_exit_price: dict = {}

    # 真实策略回测 — 入场/退出价用 log 代表价 + 3% 止损 + 连续熔断
    trades = run_backtest(
        close_d, high_d, low_d, upper_band, lower_band,
        regime, rv_pctile, gld_1h=gld_1h,
        start_date=pd.Timestamp(today_sgt) - timedelta(days=180),
        entry_log_lookup=_worst_buy_lookup,
        exit_log_lookup=_worst_exit_lookup,
        asset=asset_key)

    def _log_price(d, side):
        """从 log 取该日代表价; 无记录返回 None."""
        lk = _worst_buy_lookup if side == "BUY" else _worst_exit_lookup
        if len(lk) == 0:
            return None
        d_norm = pd.Timestamp(d).normalize()
        if d_norm not in lk.index:
            return None
        return float(lk.loc[d_norm, "price"])

    def _log_n_triggers(d, side):
        lk = _worst_buy_lookup if side == "BUY" else _worst_exit_lookup
        if len(lk) == 0:
            return 0
        d_norm = pd.Timestamp(d).normalize()
        if d_norm not in lk.index:
            return 0
        return int(lk.loc[d_norm, "n_triggers"])

    if asset_key == "SLV":
        _slv_oos = os.path.join(_intra_cfg["data_root"], "models",
                                "dl_range_slv_oos.parquet")
        range_df = pd.read_parquet(_slv_oos) if os.path.exists(_slv_oos) \
            else load_oos_predictions(_intra_cfg)
    else:
        range_df = load_oos_predictions(_intra_cfg)
    last_date = bp_dates[-1]
    last_close = close_d.get(last_date, 0)
    last_regime = regime.get(last_date, "?")
    last_bp = bp_s.get(last_date, 0)
    next_upper, next_lower, next_bp030, next_bp090 = \
        compute_next_day_band(close_d, range_df, bp_dates, last_date)

    # OI 微观结构修正 (仅 GLD — 期权快照来自 GLD 期权链)
    oi_adj_bp030 = oi_adj_bp090 = 0
    _cfg_oi = load_config()
    _eod_oi, _snap_oi = load_latest_eod_snapshot(_cfg_oi)
    if asset_key == "GLD" and _eod_oi is not None and next_upper > next_lower:
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

    # v3.7.62: SLV 用 si_slv_ratio (动态算), GLD 用 gc_gld_ratio
    if asset_key == "GLD":
        gc_gld_r = gc_gld_ratio if gc_gld_ratio else 10.9
    else:
        gc_gld_r = (locals().get("si_slv_ratio") if locals().get("si_slv_ratio")
                    else 1.11)
    _rt_ticker = "GC=F" if asset_key == "GLD" else "SI=F"
    rt = _get_realtime_prices(_rt_ticker)
    _cny = rt["usdcny"] if rt else (usdcny_rate if usdcny_rate else 7.0)
    _g = 31.1035

    # 判断是否有未平仓: 看最后一笔回测交易
    has_open_position = False
    entry_price_open = peak_open = pullback_stop = 0
    if trades:
        # 最后一笔可能是活跃仓 (exit_date=None) — 跳到最后一笔已平仓
        closed_trades = [t for t in trades
                          if t.get("exit_date") is not None
                          and not t.get("active", False)]
        if closed_trades:
            last_trade = closed_trades[-1]
            # 最后一笔已平仓 → 检查之后是否有新买入信号
            buy_after_last = sig_df[
                (sig_df["buy_signal"]) &
                (sig_df.index > last_trade["exit_date"])
            ]
        else:
            # 没有已平仓的: 第一笔就活跃, 看全部 buy 信号
            buy_after_last = sig_df[sig_df["buy_signal"]]
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
        # 判断当前信号
        # v3.7.53: 数据过期时, _raw_sig 强制清空, 不显示历史信号
        _ld_raw = pd.Timestamp(last_date)
        _stale_raw = (pd.Timestamp(today_sgt) - _ld_raw.normalize()).days > 1
        if last_date in sig_df.index and not _stale_raw:
            _raw_sig = sig_df.loc[last_date]["signal_text"]
        else:
            _raw_sig = ""
        _has_open_buy = "BUY" in _raw_sig or "SELL PUT" in _raw_sig
        # v3.7.47: 推荐仓位倍数 (sizing 来自 dir_indicators)
        _sizing = 1.0
        _sizing_reasons = ""
        if last_date in sig_df.index and "sizing" in sig_df.columns:
            _sizing = sig_df.loc[last_date].get("sizing", 1.0)
            _sizing_reasons = sig_df.loc[last_date].get("sizing_reasons", "")

        if bp_est < 0.30:
            zone = "看多"
            zone_icon_v = "🟢"
        elif bp_est > 0.90:
            zone = "看空/止盈"
            zone_icon_v = "🔴"
        elif _has_open_buy:
            zone = "持仓中"
            zone_icon_v = "🟡"
        else:
            zone = "观望"
            zone_icon_v = "⚪"
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

        _is_gold = asset_key == "GLD"
        _spot_label = "伦敦金" if _is_gold else "伦敦银"
        _shfe_label = "沪金" if _is_gold else "沪银"
        _etf_label = "GLD" if _is_gold else "SLV"
        _price_fmt = ",.0f" if _is_gold else ",.2f"

        # 直接用实时期货价格 (不用 ETF 换算)
        # gc_now 已经是 GC=F 或 SI=F 的真实价格
        if gc_now > 0 and last_close > 0:
            _spot_ratio = gc_now / last_close  # 实时期货/ETF比
        else:
            _spot_ratio = gc_gld_r

        _buy_spot = eff_bp030 * _spot_ratio
        _exit_spot = eff_bp090 * _spot_ratio
        _buy_shfe = _buy_spot * _cny / _g
        _exit_shfe = _exit_spot * _cny / _g

        # ── 状态条: 今日窗口 / 盘中实时 / 持仓 / Regime / RV ──
        # v3.7.54: 修语义 — 窗口"已开启"应该指"当前实时 bp < 0.30 可入场",
        # 而不是"今日 day-low 触过" (因 pre-market 触底但盘中已反弹的情况误导)
        _today_row = sig_df.loc[last_date] \
            if (last_date in sig_df.index and not _stale_raw) else None
        _bp_low_today = (float(_today_row["bp_low"])
                         if _today_row is not None
                         and "bp_low" in _today_row.index else None)
        # 用实时 bp_est 判断窗口当前状态
        _window_open = (bp_est is not None and bp_est < 0.30)
        # 历史是否触过 (信息性, 不等于可入场)
        _window_touched = (_bp_low_today is not None
                            and _bp_low_today < 0.30)

        # 盘中实时: 用实时 bp_est + 最近 1h Stoch RSI 状态
        # 加载日 log 看今天是否已有触发
        _intra_today_n = 0
        if len(_intra_log_asset) > 0:
            _intra_today_n = (
                _intra_log_asset["date"]
                == pd.Timestamp(today_sgt)
            ).sum()
        if gc_now > 0 and bp_est < 0.30 and _intra_today_n > 0:
            _intra_state, _intra_emo = "已触发可入场", "🟢"
        elif gc_now > 0 and bp_est < 0.30:
            _intra_state, _intra_emo = "已开窗等确认", "🟡"
        elif gc_now > 0 and bp_est > 0.90:
            _intra_state, _intra_emo = "可平仓", "🔴"
        elif gc_now > 0:
            _intra_state, _intra_emo = "观望", "⚪"
        else:
            _intra_state, _intra_emo = "实时未连", "⚫"

        _rv_pct_today = float(rv_pctile.get(last_date, 0))

        # 波动率信号 (做多 vs 做空) — v3.7.50 启用 tech-score 模式
        from core.events import (detect_straddle_signal as _dsv_long,
                                  detect_short_vol_signal as _dsv_short)
        try:
            _rv_series_sb = features["rv_10d"] \
                if "rv_10d" in features.columns \
                else pd.Series(20, index=features.index)
        except Exception:
            _rv_series_sb = pd.Series(20, index=close_d.index)
        # bug fix: 之前没传 close/high/low, 一直走旧 RV+事件 score 模式
        _vol_long_today = _dsv_long(_rv_series_sb, pd.DatetimeIndex([last_date]),
                                       rv_pctile=rv_pctile,
                                       close=close_d, high=high_d, low=low_d,
                                       asset=asset_key)
        _vol_short_today = _dsv_short(_rv_series_sb, rv_pctile,
                                         pd.DatetimeIndex([last_date]),
                                         regime=regime,
                                         close=close_d, high=high_d, low=low_d,
                                         asset=asset_key)
        _vlong_sig = (_vol_long_today["straddle_signal"].iloc[0]
                      if len(_vol_long_today) > 0 else False)
        _vshort_sig = (_vol_short_today["short_vol_signal"].iloc[0]
                       if len(_vol_short_today) > 0 else False)
        _vlong_score = (_vol_long_today["straddle_score"].iloc[0]
                        if len(_vol_long_today) > 0 else 0)
        _vshort_score = (_vol_short_today["short_vol_score"].iloc[0]
                         if len(_vol_short_today) > 0 else 0)

        # v3.7.50 STRADDLE sizing (实证 GLD 73% / SLV 70% 胜率 score≥6)
        # score 6 = 1× / 7 = 1.5× / 8+ = 2× (累计 +125% vs 单切 +68%)
        def _vol_sizing(s):
            if s >= 8: return "2×"
            if s >= 7: return "1.5×"
            if s >= 6: return "1×"
            return None
        _long_sz = _vol_sizing(_vlong_score)
        _short_sz = _vol_sizing(_vshort_score)

        if _vlong_sig and _vshort_sig:
            _vol_label = ("↑做多波动率" if _vlong_score >= _vshort_score
                          else "↓做空波动率")
            _vol_emo = "🟣"
            sz = _long_sz if _vlong_score >= _vshort_score else _short_sz
            _vol_delta = f"L{_vlong_score} / S{_vshort_score} | 仓 {sz or '—'}"
        elif _vlong_sig:
            _vol_label, _vol_emo = "↑做多波动率", "🟣"
            _vol_delta = f"score={_vlong_score} | 仓 {_long_sz or '—'}"
        elif _vshort_sig:
            _vol_label, _vol_emo = "↓做空波动率", "🟠"
            _vol_delta = f"score={_vshort_score} | 仓 {_short_sz or '—'}"
        else:
            _vol_label, _vol_emo = "中性", "⚪"
            _vol_delta = f"L{_vlong_score} / S{_vshort_score} (未触发≥6)"

        # ── 信号时效面板 + 关键事件倒计时 (v3.7.19) ──
        # US 期权时段 SGT: 21:30 ~ 04:00 (次日)
        # v3.7.75: 全部用美东时间 (yfinance 数据 TZ + dashboard 一致)
        from datetime import datetime, timezone, timedelta as _td
        _now_et = pd.Timestamp.now(tz="America/New_York")
        _now_h = _now_et.hour + _now_et.minute / 60.0
        _now_naive = _now_et.tz_localize(None)

        _us_open = 9.5  # 09:30 ET
        _us_close = 16.0  # 16:00 ET
        _is_us_session = _us_open <= _now_h < _us_close
        if _is_us_session:
            _session_state = "🟢 US 期权时段中 (可交易)"
        elif _now_h < _us_open:
            _hours_to_next = _us_open - _now_h
            _h, _m = int(_hours_to_next), int((_hours_to_next % 1) * 60)
            _session_state = f"⏳ 距 US 开盘 {_h}h{_m}m (ET 09:30)"
        else:
            _hours_to_next = (24 + _us_open) - _now_h
            _h, _m = int(_hours_to_next), int((_hours_to_next % 1) * 60)
            _session_state = f"⏳ 距 US 开盘 {_h}h{_m}m"

        # v3.7.75: 信号时效 — 全部美东时间; last_date 是 US 交易日
        # US close = 16:00 ET, 信号约 16:05 ET 生成
        _signal_gen_time = pd.Timestamp(last_date).tz_localize(None) \
            + _td(hours=16, minutes=5)
        _signal_age_h = (_now_naive - _signal_gen_time).total_seconds() / 3600
        _signal_label = (
            f"⏰ **信号时效** (全部美东时间 ET): 基于 **{last_date.date()} US close** "
            f"(约 {_signal_age_h:.0f}h 前) | "
            f"{_session_state} | "
            f"**适用于下一个 US 盘中** | "
            f"入场前请刷新核对 RV/bp_low/IV"
        )
        st.info(_signal_label)

        sb1, sb2, sb3, sb4, sb5 = st.columns(5)
        with sb1:
            # v3.7.54/55: 三态 — 可入场 / 已触过 (含 pre-market 反弹) / 未开启
            # 含日内 + 1h pre-market 是否触过 0.30 buy zone
            if _window_open:
                _w_label, _w_delta = "✅ 可入场", f"实时 bp={bp_est:.2f} < 0.30"
            elif _window_touched:
                # 区分: 是日线 daily Low 触过, 还是 pre-market 1h 触过
                _w_label = "⚠️ 已触过 (无效)"
                _w_delta = (f"今日触底 bp={_bp_low_today:.2f} 但已反弹"
                            f" → 现 bp={bp_est:.2f}, 不可入场,"
                            f" 等盘中回踩 + 技术确认")
            else:
                _w_label = "未开启"
                _w_delta = (f"现 bp={bp_est:.2f} (需 < 0.30)"
                            if bp_est else "—")
            st.metric("今日窗口", _w_label, delta=_w_delta)
        with sb2:
            st.metric("盘中实时", f"{_intra_emo} {_intra_state}",
                      delta=f"实时 bp≈{bp_est:.2f} | "
                            f"今日触发 {_intra_today_n} 次")
        with sb3:
            st.metric("波动率信号", f"{_vol_emo} {_vol_label}",
                      delta=_vol_delta)
        with sb4:
            st.metric("Regime", last_regime,
                      delta=f"数据至 {last_date.date()}")
        with sb5:
            _rv_zone = ("低位" if _rv_pct_today < 0.30
                        else ("高位" if _rv_pct_today > 0.85
                              else "正常"))
            st.metric("RV %tile", f"{_rv_pct_today:.0%}",
                      delta=_rv_zone)
        st.divider()

        oi_tag = " (OI修正)" if oi_adj_bp030 > 0 else ""
        st.markdown('<div class="signal-box">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"看多 <{oi_tag}", f"${_buy_spot:{_price_fmt}}",
                      delta=f"{_etf_label} < ${eff_bp030:.2f} | {_shfe_label} < ¥{_buy_shfe:.1f}")
        with c2:
            st.metric(f"看空/止盈 >{oi_tag}", f"${_exit_spot:{_price_fmt}}",
                      delta=f"{_etf_label} > ${eff_bp090:.2f} | {_shfe_label} > ¥{_exit_shfe:.1f}")
        with c3:
            # v3.7.62: session-aware 工具路由 — 默认期权, 非 US session 用期货补窗外
            # US 期权 regular session: SGT 21:30 ~ 04:00 (= US 9:30-16:00 EDT)
            # 之外 (盘前/盘后/亚欧时段): 期权流动性差, 用期货 24h 抓机会
            _fut_label = "GC=F" if asset_key == "GLD" else "SI=F"
            _opt_label = "GLD" if asset_key == "GLD" else "SLV"
            if _is_us_session:
                _sig_map = {
                    "BUY CALL": f"🎯 {_opt_label} Buy Call (期权 US session)",
                    "SELL PUT": f"🎯 {_opt_label} Sell Put (期权 推 100%)",
                    "EXIT": f"平仓 / {_opt_label} 做空",
                    "BUY CALL + EXIT": f"{_opt_label} Buy Call (有退出)",
                    "SELL PUT + EXIT": f"{_opt_label} Sell Put (有退出)",
                }
            else:
                _sig_map = {
                    "BUY CALL": f"📈 期货多头 {_fut_label} (期权关闭, 24h 抓窗外机会)",
                    "SELL PUT": f"📈 期货多头 {_fut_label} (期权关闭, sell put 不可代用)",
                    "EXIT": f"📉 期货空头 {_fut_label} (期权关闭)",
                    "BUY CALL + EXIT": f"期货多头 {_fut_label} (有退出)",
                    "SELL PUT + EXIT": f"期货多头 {_fut_label} (有退出)",
                }
            # 数据过期检测: last_date 距今 > 1 个交易日 视为过期
            _ld = pd.Timestamp(last_date)
            _data_age = (pd.Timestamp(today_sgt) - _ld.normalize()).days
            _is_stale = _data_age > 1

            # v3.7.54: 信号 = 当前可执行 (实时 bp_est < 0.30)
            # 历史触过 (今日 pre-market 跌过) → 显"已触过", 不推工具入场
            if _is_stale:
                sig_text = "数据过期"
                _delta_str = (f"末数据 {_ld.date()} ({_data_age}d ago) — "
                              f"重启 dashboard 触发数据刷新")
            elif _raw_sig and bp_est is not None and bp_est < 0.30:
                # 当下 bp 仍在 buy zone — 工具推荐有效
                sig_text = _sig_map.get(_raw_sig, _raw_sig)
                _sizing_tag = ""
                if _sizing > 1.0 and _has_open_buy:
                    _sizing_tag = f" | 仓位 {_sizing:.0f}× ({_sizing_reasons})"
                _delta_str = (f"Regime: {last_regime} | 实时 bp={bp_est:.3f} | "
                              f"RV={rv_pctile.get(last_date,0):.0%}{_sizing_tag}")
            elif _raw_sig:
                # 今日 day-low 触过但已反弹 — 不可执行
                sig_text = "今日已触过 (已反弹)"
                _delta_str = (f"曾触发 {_raw_sig} 但现 bp={bp_est:.2f} > 0.30, "
                              f"等下次回踩")
            else:
                sig_text = "今日无信号"
                _delta_str = (f"现 bp={bp_est:.3f} (需 < 0.30) | "
                              f"RV={rv_pctile.get(last_date,0):.0%} | "
                              f"持仓 / 历史见可视化")
            st.metric("当日信号", sig_text, delta=_delta_str)
        st.markdown('</div>', unsafe_allow_html=True)

        # 实时价格行 (紫色背景)
        if gc_now > 0:
            xau_est = gc_now
            shfe_est = gc_now * _cny / _g

            # 根据资产类型显示对应期货
            _is_gold_rt = asset_key == "GLD"
            _spot_label_rt = "伦敦金" if _is_gold_rt else "伦敦银"
            _etf_label_rt = "GLD" if _is_gold_rt else "SLV"
            _pfmt = ".1f" if _is_gold_rt else ".2f"

            st.markdown('<div class="price-box">', unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns([3, 2, 2, 2])
            with r1:
                st.metric(f"{zone_icon_v} {_spot_label_rt}",
                          f"${gc_now:{_pfmt}}",
                          delta=f"{zone} | {_etf_label_rt}≈${gld_est:{_pfmt}} bp≈{bp_est:.2f}")
            with r2:
                st.metric("沪金" if _is_gold_rt else "沪银",
                          f"¥{shfe_est:.2f}",
                          delta=f"CNY={_cny:.4f}")
            with r3:
                # 币安
                try:
                    from core.binance_data import fetch_binance_prices
                    _bn = fetch_binance_prices()
                    if _bn:
                        _bn_key = "xau_price" if _is_gold else "xag_price"
                        _bn_chg = "xau_change" if _is_gold else "xag_change"
                        if _bn_key in _bn:
                            _bn_fmt = ".1f" if _is_gold_rt else ".2f"
                            st.metric("币安合约",
                                      f"${_bn[_bn_key]:{_bn_fmt}}",
                                      delta=f"{_bn.get(_bn_chg,0):+.1f}% 24h")
                except Exception:
                    st.metric("币安", "—")
            with r4:
                st.metric("时间", ts if ts else "—",
                          delta=f"{last_date.date()}")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.caption(f"实时数据未获取 | {asset_key} 收盘 ${last_close:.2f} ({last_date.date()})")

    # ── 市场分析 ──
    st.divider()
    from core.events import days_to_next_event, detect_straddle_signal
    from core.data import load_features
    features = load_features(load_config())
    features = features.reindex(close_d.index).ffill()

    # 近期事件
    d_fomc, _, fomc_d = days_to_next_event(last_date, "FOMC")
    d_opex, _, opex_d = days_to_next_event(last_date, "OPEX")
    d_nfp, _, nfp_d = days_to_next_event(last_date, "NFP")

    # 宏观指标
    rv = features["rv_10d"].get(last_date, 0) if "rv_10d" in features.columns else 0
    dxy = features["dxy_ret_5d"].get(last_date, 0) if "dxy_ret_5d" in features.columns else 0
    vix = features["vix_level"].get(last_date, 0) if "vix_level" in features.columns else 0
    real_y = features["real_yield_10y"].get(last_date, 0) if "real_yield_10y" in features.columns else 0
    fed_r = features["fed_funds_rate"].get(last_date, 0) if "fed_funds_rate" in features.columns else 0

    # Straddle 检测
    rv_s = features["rv_10d"] if "rv_10d" in features.columns else pd.Series(20, index=features.index)
    # v3.7.47: 传 close/high/low 启用技术指标 score 模式
    straddle_today = detect_straddle_signal(
        rv_s, pd.DatetimeIndex([last_date]), rv_pctile=rv_pctile,
        close=close_d, high=high_d, low=low_d,
        asset=asset_key)
    is_straddle = straddle_today["straddle_signal"].iloc[0] if len(straddle_today) > 0 else False
    straddle_reason = straddle_today["straddle_reason"].iloc[0] if is_straddle else ""

    with st.expander("市场环境分析", expanded=True):
        col_ev, col_macro = st.columns(2)
        with col_ev:
            st.markdown("**近期事件 (倒计时按公告时间)**")
            # 倒计时按 SGT 公告时间, 不只是事件日
            from datetime import timedelta as _td
            from core.events import get_all_events
            _now_naive_lite = pd.Timestamp.now(
                tz="America/New_York").tz_localize(None)
            _ev_list = get_all_events(
                _now_naive_lite.normalize().strftime("%Y-%m-%d"),
                (_now_naive_lite + _td(days=14)).strftime("%Y-%m-%d"),
                asset=("gold" if asset_key == "GLD" else "silver"))
            if _ev_list:
                _ev_cards = []
                for ev_d, ev_t, ev_l in _ev_list[:5]:
                    # SGT 公告时刻
                    if ev_t == "FOMC":
                        ann = pd.Timestamp(ev_d).tz_localize(None) + _td(hours=2)
                    elif ev_t == "NFP":
                        ann = pd.Timestamp(ev_d).tz_localize(None) + _td(hours=20, minutes=30)
                    elif ev_t == "OPEX":
                        ann = pd.Timestamp(ev_d).tz_localize(None) + _td(hours=4)
                    else:
                        ann = pd.Timestamp(ev_d).tz_localize(None) + _td(hours=4)
                    delta = (ann - _now_naive_lite).total_seconds() / 3600
                    if delta < 0:
                        continue
                    days = int(delta // 24)
                    hrs = int(delta % 24)
                    mins = int((delta * 60) % 60)
                    cd_str = (f"{days}d {hrs}h{mins}m" if days > 0
                              else f"{hrs}h{mins}m")
                    _ev_cards.append((ev_l, cd_str, ann.strftime("%m/%d %H:%M")))
                for label, cd, ann_str in _ev_cards:
                    st.markdown(f"- **{label}** — 倒计时 {cd} (SGT {ann_str})")
            else:
                st.markdown("- 未来 14 天无重大事件")

            if is_straddle:
                st.warning(f"Straddle 信号: {straddle_reason}")
            elif min(d_fomc, d_opex) <= 3:
                st.info(f"临近事件日 — 注意波动率变化")

        with col_macro:
            st.markdown("**宏观指标**")
            st.markdown(f"""
- RV(10d): **{rv:.1f}%** {'(低位)' if rv < 20 else '(正常)' if rv < 35 else '(高位)'}
- VIX: **{vix:.1f}**
- DXY 5d: **{dxy*100:+.2f}%** {'(美元走强→金价承压)' if dxy > 0.005 else '(美元走弱→金价利好)' if dxy < -0.005 else '(中性)'}
- 实际利率: **{real_y:.2f}%** {'(偏高→压制金价)' if real_y > 2.0 else ''}
- 联邦基金: **{fed_r:.2f}%**
""")

    # ── 信号历史图 ──
    st.divider()
    # v3.7.79: K 线精度选项 (yfinance 限制: 1m=7d, 5m/15m=60d, 1h=730d)
    # 未来实盘可扩展到 1s (IBKR/Polygon 付费 API)
    _gran = st.sidebar.radio("主图粒度",
                              ["盘中 (1-14 天 高精度)", "日线 (历史)"],
                              index=0)
    _is_1h_view = _gran.startswith("盘中")
    if _is_1h_view:
        # K 线精度: 默认 5m (v3.7.81 回测 EU=38 vs 15m=27, 触发更准)
        _intraday_interval = st.sidebar.selectbox(
            "K 线精度",
            ["1m (近 7 天)", "5m (近 60 天, 默认)", "15m (近 60 天)",
             "30m (近 60 天)", "1h (近 730 天)"],
            index=1,  # 5m 默认
            help="yfinance 限制. 未来实盘可扩 1s (IBKR/Polygon 付费)")
        # 解析 interval code
        _interval_code = _intraday_interval.split()[0]  # '1m','5m','15m','30m','1h'
        # 默认范围: 14 天 (用户偏好两周内)
        _max_days = {"1m": 7, "5m": 60, "15m": 60,
                      "30m": 60, "1h": 730}.get(_interval_code, 60)
        # v3.7.80: 默认 7 天, 范围 1-14 天 (用户偏好两周内)
        _intraday_days = st.sidebar.slider(
            f"{_interval_code} 回看天数", 1, min(14, _max_days),
            min(7, _max_days))
    else:
        lookback_days = st.sidebar.slider("回看天数", 30, 180, 65)
        lookback = last_date - timedelta(days=lookback_days)
    # v3.7.85: 现在 _interval_code 已就绪, 计算 avg trigger time (按 timeframe 过滤)
    _avg_tf_match = ((f"{_interval_code.replace('m','')}m"
                        if _interval_code != "1h" else "60m")
                       if _is_1h_view else "60m")
    _avg_buy_time = _avg_trigger_time("BUY", _avg_tf_match)
    _avg_exit_time = _avg_trigger_time("EXIT", _avg_tf_match)
    # v3.7.86: 同步算价
    _avg_buy_price = _avg_trigger_price("BUY", _avg_tf_match)
    _avg_exit_price = _avg_trigger_price("EXIT", _avg_tf_match)
    # v3.7.84: 主图数据源改 COMEX 期货 (GC=F/SI=F) — 23h 全球夜盘, 真伦敦金价
    # ETF (GLD/SLV) 只 US session 6.5h, 缺亚欧夜盘 — 用户诉求显示 24h
    # 实现: 拉 GC=F (gold scale $3800) → 除以 _viz_ratio → ETF-等价 scale ($400)
    #      下游 × _r 逻辑不变, 仅新增 overnight bars
    _kline_is_futures = False  # v3.7.84: 控制 dirty filter 跳过
    if _is_1h_view and _interval_code != "1h":
        _yf_ticker_for_chart = "GC=F" if asset_key == "GLD" else "SI=F"
        _intraday_data = _fetch_intraday(_yf_ticker_for_chart,
                                            _interval_code,
                                            _intraday_days)
        if _intraday_data is not None and len(_intraday_data) > 0:
            # 计算 ratio: 用最近 close 推算 (避免 _viz_ratio 还没初始化)
            _ratio_probe = (_intraday_data["Close"].iloc[-1]
                              / last_close if last_close > 0 else 1.0)
            _intraday_data = _intraday_data.copy()
            for _col in ["Open", "High", "Low", "Close"]:
                _intraday_data[_col] = _intraday_data[_col] / _ratio_probe
            gld_1h = _intraday_data
            _kline_is_futures = True
            st.sidebar.caption(
                f"✓ {_yf_ticker_for_chart} {_interval_code}: "
                f"{len(gld_1h)} 行 (23h 全球, ratio {_ratio_probe:.2f})")
        else:
            st.sidebar.caption(
                f"⚠ {_yf_ticker_for_chart} 拉取失败, 退回 ETF (US-only)")

    if _is_1h_view:
        viz_dates = close_d.index[
            (close_d.index >= last_date - timedelta(days=_intraday_days+5))
            & (close_d.index <= last_date)]
    else:
        viz_dates = close_d.index[(close_d.index >= lookback) & (close_d.index <= last_date)]
    sig_viz = sig_df.reindex(viz_dates).dropna(subset=["close"])

    # v3.7.78: 重排 subplot 顺序 — 主图 / 1h K线 / 1h Stoch / 15m Stoch / Squeeze
    # (原 v3.7.25: 主图 / 1h Stoch / 15m Stoch / 1h K线 / Squeeze)
    # K 线紧接主图后, stoch 子图集中放后段 (用户偏好)
    fig, (ax, ax_kline, ax_stoch_1h, ax_stoch_15m, ax_sq_main) = plt.subplots(
        5, 1, figsize=(18, 18), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1, 1, 1], "hspace": 0.08})

    # ── 价位换算: 主图用伦敦金/伦敦银 (现货/期货), 不再用 ETF 价位 ──
    _viz_ticker = "GC=F" if asset_key == "GLD" else "SI=F"
    _viz_rt = _get_realtime_prices(_viz_ticker)
    if _viz_rt and _viz_rt.get("gc_price", 0) > 0 and last_close > 0:
        _viz_ratio = _viz_rt["gc_price"] / last_close
    elif gc_gld_ratio:
        _viz_ratio = gc_gld_ratio
    else:
        _viz_ratio = 1.0
    _viz_spot_label = "伦敦金" if asset_key == "GLD" else "伦敦银"
    _viz_unit = "USD/oz"
    _r = _viz_ratio  # 简写

    if _is_1h_view and gld_1h is not None and len(gld_1h) > 0:
        # ─── 1h 主图: 近 N 天 (默认 3) ───
        _cutoff = last_date - timedelta(days=_intraday_days)
        _1h = gld_1h[gld_1h.index >= _cutoff].copy()
        if len(_1h) == 0:
            _1h = gld_1h.iloc[-_intraday_days*24:].copy()
        # v3.7.65: 过滤 yfinance prepost 脏 1h bar (仅 ETF 源, 期货数据干净)
        # v3.7.84: 期货源跳过 — overnight 价格合法超 ETF daily L/H (亚欧夜盘真行情)
        if not _kline_is_futures:
            _dates_n = _1h.index.normalize()
            _daily_lo_map = low_d.reindex(_dates_n.unique())
            _daily_hi_map = high_d.reindex(_dates_n.unique())
            _1h_dlow = _1h.index.to_series().apply(
                lambda t: _daily_lo_map.get(t.normalize(), float("nan")))
            _1h_dhigh = _1h.index.to_series().apply(
                lambda t: _daily_hi_map.get(t.normalize(), float("nan")))
            _1h_dlow = _1h_dlow.fillna(_1h["Close"])
            _1h_dhigh = _1h_dhigh.fillna(_1h["Close"])
            _1h["Low"] = _1h["Low"].clip(lower=_1h_dlow * 0.997)
            _1h["High"] = _1h["High"].clip(upper=_1h_dhigh * 1.003)
            _1h["Close"] = _1h["Close"].clip(
                lower=_1h_dlow * 0.997, upper=_1h_dhigh * 1.003)
            _1h["Open"] = _1h["Open"].clip(
                lower=_1h_dlow * 0.997, upper=_1h_dhigh * 1.003)
        plot_ts = list(_1h.index)
        ts2i = {ts: i for i, ts in enumerate(plot_ts)}
        # xi() 接受 daily 日期, 找到当日第一个 1h timestamp
        def xi(d):
            d_norm = pd.Timestamp(d).normalize()
            for i, ts in enumerate(plot_ts):
                if ts.normalize() == d_norm:
                    return i
            return None
        def xi_arr(dates): return [xi(d) for d in dates if xi(d) is not None]
        # v3.7.74: 时间显示 = 美国东部时间 (EDT/EST, yfinance 1h 原 TZ)
        _show_hours = _intraday_days <= 5
        def _fmt_tick(x, pos):
            idx = int(round(x))
            if 0 <= idx < len(plot_ts):
                ts = plot_ts[idx]
                if _show_hours:
                    return ts.strftime("%m/%d %H:%M") + " ET"
                # v3.7.84: 日界线 label 加时间, 防误读为 0 点
                # (RTH only: 每天起点是 09:30, 不是 midnight)
                if idx == 0 or plot_ts[idx-1].date() != ts.date():
                    return ts.strftime("%m/%d") + f"\n{ts.strftime('%H:%M')} ET"
                # 收盘前最后一根加 close 时间标记
                if (idx + 1 < len(plot_ts)
                        and plot_ts[idx+1].date() != ts.date()):
                    return ts.strftime("%H:%M") + " ET"
                return ""
            return ""
        # 兼容旧 plot_dates 引用 (后续部分用 plot_dates 算 xlim 等)
        plot_dates = plot_ts
        d2i = ts2i

        # 1h 收盘线 (×_r)
        ax.plot(range(len(_1h)), _1h["Close"].values * _r,
                color="black", lw=1.5, zorder=3, label="1h 收盘")
        # 1h 高低区 (浅灰)
        ax.fill_between(range(len(_1h)),
                          _1h["Low"].values * _r,
                          _1h["High"].values * _r,
                          alpha=0.10, color="gray", zorder=1)

        # ── Y 轴自动 zoom 到实际 1h 价格范围 (避免 Band 撑开导致 1h 波动看起来扁) ──
        _y_low = (_1h["Low"].min()) * _r
        _y_high = (_1h["High"].max()) * _r
        _y_pad = (_y_high - _y_low) * 0.15  # 15% 边距
        ax.set_ylim(_y_low - _y_pad, _y_high + _y_pad)

        # 日 Band 投影到 1h: 每天画水平段
        ub_plot = upper_band.reindex(viz_dates).dropna()
        lb_plot = lower_band.reindex(viz_dates).dropna()
        cidx = ub_plot.index.intersection(lb_plot.index)
        for d in cidx:
            d_norm = pd.Timestamp(d).normalize()
            in_day = [i for i, ts in enumerate(plot_ts)
                      if ts.normalize() == d_norm]
            if not in_day:
                continue
            x_seg = [in_day[0], in_day[-1]]
            u_v, l_v = ub_plot[d] * _r, lb_plot[d] * _r
            bp030_v = (lb_plot[d] + 0.30 * (ub_plot[d] - lb_plot[d])) * _r
            bp090_v = (lb_plot[d] + 0.90 * (ub_plot[d] - lb_plot[d])) * _r
            # Band 上下界
            ax.plot(x_seg, [u_v, u_v], color="green", lw=1.2, alpha=0.6, zorder=2)
            ax.plot(x_seg, [l_v, l_v], color="magenta", lw=1.2, alpha=0.6, zorder=2)
            ax.fill_between(x_seg, [l_v, l_v], [u_v, u_v],
                              alpha=0.05, color="green", zorder=1)
            # bp030 / bp090 阈值线
            ax.plot(x_seg, [bp030_v, bp030_v],
                     color="#2196F3", lw=1.0, ls="--", alpha=0.7, zorder=2)
            ax.plot(x_seg, [bp090_v, bp090_v],
                     color="#F44336", lw=1.0, ls="--", alpha=0.7, zorder=2)
    else:
        # ─── 日线主图 (旧版逻辑) ───
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
        ax.plot(xi_arr(cl_plot.index), cl_plot.values * _r, "k-", lw=1.8, zorder=3)
        hi_plot = high_d.reindex(viz_dates).dropna()
        lo_plot = low_d.reindex(viz_dates).dropna()
        hl_common = hi_plot.index.intersection(lo_plot.index)
        if len(hl_common) > 0:
            ax.fill_between(xi_arr(hl_common),
                             lo_plot[hl_common].values * _r,
                             hi_plot[hl_common].values * _r,
                             alpha=0.08, color="gray")
        # Band
        ub_plot = upper_band.reindex(viz_dates).dropna()
        lb_plot = lower_band.reindex(viz_dates).dropna()
        cidx = ub_plot.index.intersection(lb_plot.index)
        if len(cidx) > 0:
            ax.fill_between(xi_arr(cidx), lb_plot[cidx].values * _r,
                             ub_plot[cidx].values * _r,
                             alpha=0.06, color="green")
            ax.plot(xi_arr(cidx), ub_plot[cidx].values * _r,
                    color="green", lw=1, alpha=0.5)
            ax.plot(xi_arr(cidx), lb_plot[cidx].values * _r,
                    color="magenta", lw=1, alpha=0.5)
            bp030_line = lb_plot[cidx] + 0.30 * (ub_plot[cidx] - lb_plot[cidx])
            bp090_line = lb_plot[cidx] + 0.90 * (ub_plot[cidx] - lb_plot[cidx])
            ax.plot(xi_arr(cidx), bp030_line.values * _r,
                    color="#2196F3", lw=0.8, ls="--", alpha=0.5)
            ax.plot(xi_arr(cidx), bp090_line.values * _r,
                    color="#F44336", lw=0.8, ls="--", alpha=0.5)

    # 统一信号标注 (每天只标一个最优推荐) — 用共享 dedupe
    from core.events import (get_all_events,
                              detect_straddle_signal as _dst,
                              detect_short_vol_signal as _dsv)
    from core.strategy_selector import (
        build_unified_signals as _bus,
        dedupe_unified as _dedupe,
    )

    _straddle_viz = _dst(rv_s, viz_dates, rv_pctile=rv_pctile, asset=asset_key)
    _short_vol_viz = _dsv(rv_s, rv_pctile, viz_dates, regime=regime)
    _unified_viz_raw = _bus(sig_df, _straddle_viz, close_d, high_d, low_d,
                             short_vol_df=_short_vol_viz)

    def _intra_log_price(d, side):
        return _log_price(d, side)
    _unified_viz = _dedupe(_unified_viz_raw, close_d,
                            log_price_fn=_intra_log_price,
                            low_d=low_d)

    # v3.7.37: 图例始终显示全部 marker 类型, 不受当前窗口信号影响
    _sig_colors = {
        "BUY CALL": ("#2196F3", "^"),    # 蓝 ▲
        "SELL PUT": ("#FF9800", "^"),    # 橙 ▲ (方向性都用 ▲)
        "EXIT": ("#F44336", "v"),        # 红 ▼
        "STRADDLE": ("#FFD700", "*"),    # 金 ★ (做多波动率)
        "SHORT_VOL": ("#FF6F00", "P"),   # 橘 ✚ (做空波动率, P = plus 十字)
    }
    # 加 dummy scatter 让所有 5 个 marker 都进 legend
    _legend_labels = {
        "BUY CALL": "▲ BUY CALL (低 RV 做多)",
        "SELL PUT": "▲ SELL PUT (高 RV 做多, 收 IV)",
        "EXIT": "▼ EXIT (退出)",
        "STRADDLE": "★ STRADDLE (做多波动率)",
        "SHORT_VOL": "✚ SHORT_VOL Iron Condor (做空波动率)",
    }
    for _key in ["BUY CALL", "SELL PUT", "EXIT", "STRADDLE", "SHORT_VOL"]:
        _c, _m = _sig_colors[_key]
        # 透明且超出范围, 只为 legend 占位
        ax.scatter([-100], [0], marker=_m, s=120, color=_c,
                    edgecolors="black", lw=0.7,
                    label=_legend_labels[_key])
    _legend_added = set()

    # v3.7.72: helper — 美股开盘 09:30 EDT 1h bar 位置 (用于 STRADDLE/SHORT_VOL)
    def _xi_open(d):
        d_norm = pd.Timestamp(d).normalize()
        cands = [(i, ts) for i, ts in enumerate(plot_ts)
                 if ts.normalize() == d_norm
                 and 9 <= (ts.hour + ts.minute/60.0) <= 11]
        if cands:
            return min(cands, key=lambda x: x[1])[0]
        return xi(d)

    # v3.7.73: helper — trigger 平均时间投影 (用于方向性 markers x 位置)
    def _xi_at(ts):
        """找最接近 ts 的 plot_ts 索引位置."""
        if ts is None or pd.isna(ts):
            return None
        try:
            target_num = mdates.date2num(pd.Timestamp(ts))
            ref_nums = [mdates.date2num(t) for t in plot_ts]
            return int(np.interp(target_num, ref_nums, np.arange(len(plot_ts))))
        except Exception:
            return None

    # v3.7.70/72: 混合信号不合并; STRADDLE/SHORT_VOL 标在美股开盘时间点
    for d, r in _unified_viz.iterrows():
        if xi(d) is None:
            continue
        chosen = r["chosen"]
        entry_p = r["entry_p"]
        # 拆分 + 加未被 chosen 但触发的 vol 信号
        parts = [p.strip() for p in chosen.split("+") if p.strip()]
        if r.get("straddle_signal", False) and "STRADDLE" not in parts:
            parts.append("STRADDLE")
        if r.get("short_vol_signal", False) and "SHORT_VOL" not in parts:
            parts.append("SHORT_VOL")
        is_add = bool(r.get("is_add", False))
        for j, part in enumerate(parts):
            color, marker = _sig_colors.get(part, ("gray", "o"))
            size = (200 if part in ("STRADDLE", "SHORT_VOL")
                    else (160 if is_add else 120))
            edge = "purple" if is_add and part in ("BUY CALL", "SELL PUT") \
                   else "black"
            # v3.7.73/77: x 位置精准
            # y: 方向性 用 entry_p (= dedupe 平均价)
            #    STRADDLE/SHORT_VOL 用 daily close (不该跟 entry_p 一致)
            if part in ("STRADDLE", "SHORT_VOL"):
                _xi_target = _xi_open(d)
                _yp = float(close_d.get(d, entry_p))  # daily close
            elif part in ("BUY CALL", "SELL PUT"):
                # v3.7.85: 只有 active timeframe 有 trigger 时显示, 跟副图一致
                _avg_t = _avg_buy_time.get(pd.Timestamp(d))
                if _avg_t is None and _is_1h_view:
                    continue  # 当前 interval 当天没触发, 跳过 marker
                _xi_target = _xi_at(_avg_t) if _avg_t else xi(d)
                _yp = entry_p
            elif part == "EXIT":
                # v3.7.89: EXIT 不在这里画, 走下方独立 log-driven loop
                continue
            else:
                _xi_target = xi(d)
                _yp = entry_p
            if _xi_target is None:
                continue
            y_offset = j * (_yp * 0.003)
            ax.scatter([_xi_target], [(_yp + y_offset) * _r],
                        marker=marker, s=size, color=color,
                        edgecolors=edge, lw=1.0, zorder=6)
    # v3.7.89/90: EXIT marker 单一来源 — intraday log 直读
    # 日线模式: x = xi(date), y = trigger price (聚合该日 dedupe 后均价)
    if len(_intra_log_asset) and not _is_1h_view:
        from core.intraday_triggers import dedupe_intraday as _dd_exit_d
        _ex_d = _intra_log_asset[_intra_log_asset["side"] == "EXIT"]
        for _d_e, _g_e in _ex_d.groupby("date"):
            _x_d = xi(pd.Timestamp(_d_e))
            if _x_d is None: continue
            _dd_d = _dd_exit_d(_g_e, side="EXIT", min_drop_pct=0.3)
            if not len(_dd_d): continue
            _avg_p = float(pd.to_numeric(_dd_d["price"], errors="coerce").mean())
            ax.scatter([_x_d], [_avg_p * _r],
                        marker="v", s=140, color="#F44336",
                        edgecolors="black", lw=1.2, zorder=7,
                        label="_nolegend_")
    # 盘中模式 (1h view): x = 真实 trigger time, y = trigger price (每条 EXIT 一标)
    # v3.7.90: 加 viz_dates filter (np.interp 会把窗外 trigger clamp 到边缘错位)
    if len(_intra_log_asset) and _is_1h_view and len(plot_ts) > 0:
        from core.intraday_triggers import dedupe_intraday as _dd_exit
        _viz_lo = plot_ts[0].normalize()
        _viz_hi = plot_ts[-1].normalize()
        _ex_dates = pd.to_datetime(_intra_log_asset["date"]).dt.normalize()
        _exit_log = _intra_log_asset[
            (_intra_log_asset["side"] == "EXIT") &
            (_ex_dates >= _viz_lo) & (_ex_dates <= _viz_hi)]
        if len(_exit_log):
            for _d_exit, _grp_exit in _exit_log.groupby("date"):
                _dd_e = _dd_exit(_grp_exit, side="EXIT", min_drop_pct=0.3)
                for _, _re in _dd_e.iterrows():
                    _t_e = pd.Timestamp(_re["trigger_time"])
                    # 严格在 plot_ts 范围内才画 (防 np.interp clamp 错位)
                    if _t_e < plot_ts[0] or _t_e > plot_ts[-1]:
                        continue
                    _x_e = _xi_at(_t_e)
                    if _x_e is None: continue
                    ax.scatter([_x_e], [float(_re["price"]) * _r],
                                marker="v", s=140, color="#F44336",
                                edgecolors="black", lw=1.2, zorder=7,
                                label="_nolegend_")
    # 始终显示全部 5 类 + MIXED 边框说明 (即使当前窗口没有该类信号)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2,
               title="信号类型 (紫色边框 = MIXED 组合)")
    # 防止 dummy scatter (x=-100) 影响 X 轴自动缩放
    if len(plot_dates) > 0:
        ax.set_xlim(-0.5, len(plot_dates) - 0.5)

    # v3.7.89: 删除回测 exit 散点 (用户反馈"落在每次日期刻度线上"刺眼)
    # EXIT marker 现单一来源 = intraday log (上方独立循环), 真实时间+价格.

    # 事件日期标注 (FOMC/OPEX/NFP)
    _asset_type = "gold" if asset_key == "GLD" else "silver"
    events_in_range = get_all_events(
        viz_dates[0].strftime("%Y-%m-%d"), viz_dates[-1].strftime("%Y-%m-%d"),
        asset=_asset_type)
    ev_colors = {"FOMC": "#E91E63", "OPEX": "#FF9800", "NFP": "#3F51B5",
                  "FUT_EXP": "#795548"}
    for ev_d, ev_type, ev_label in events_in_range:
        ev_xi = xi(ev_d)
        if ev_xi is not None:
            ax.axvline(ev_xi, color=ev_colors.get(ev_type, "gray"),
                       lw=1.2, ls=":", alpha=0.5, zorder=1)
            ax.annotate(ev_label, xy=(ev_xi, ax.get_ylim()[1]),
                        xytext=(0, -8), textcoords="offset points",
                        fontsize=7, color=ev_colors.get(ev_type, "gray"),
                        fontweight="bold", ha="center", va="top")

    legend_el = [
        Line2D([0],[0], color="k", lw=1.5, label=_viz_spot_label),
        Line2D([0],[0], color="green", lw=1, alpha=0.5, label="Band"),
        Line2D([0],[0], color="#2196F3", lw=0.8, ls="--", label="Buy线"),
        Line2D([0],[0], color="#F44336", lw=0.8, ls="--", label="Exit线"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#2196F3", markersize=9, label="BUY CALL"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#FF9800", markersize=9, label="SELL PUT"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#F44336", markersize=9, label="EXIT"),
        Line2D([0],[0], marker="*", color="#FFD700", markersize=12, label="STRADDLE"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#FF6600", markersize=7, alpha=0.5, label="止盈"),
        Line2D([0],[0], color="#E91E63", lw=1, ls=":", label="FOMC"),
        Line2D([0],[0], color="#FF9800", lw=1, ls=":", label="OPEX"),
    ]
    ax.legend(handles=legend_el, loc="upper left", fontsize=6, ncol=6)
    ax.set_title(f"盘中信号 (Band + 盘中触发入场 + "
                 f"StopLoss/BandExit/Pullback + 事件日) | "
                 f"数据至 {last_date.date()} | Regime: {last_regime} | "
                 f"换算 {_viz_ratio:.4f} ({asset_key}→{_viz_spot_label})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{_viz_spot_label} ({_viz_unit})")
    ax.grid(True, alpha=0.3)
    # v3.7.28: 主图也显示 datetime 时间刻度 (顶部 + 底部都标)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    # v3.7.85: 强制每天起点都是刻度 (不论 zoom level)
    if _is_1h_view and len(plot_ts) > 1:
        _day_start_idx = [i for i, t in enumerate(plot_ts)
                            if i == 0 or plot_ts[i-1].date() != t.date()]
        ax.xaxis.set_major_locator(FixedLocator(_day_start_idx))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    ax.tick_params(axis="x", labelbottom=True, labeltop=False, rotation=0)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    # v3.7.84: 日界线竖线 (强调 overnight gap, RTH-only 拼接的真实切换)
    if _is_1h_view and len(plot_ts) > 1:
        for _i, _t in enumerate(plot_ts[1:], start=1):
            if plot_ts[_i-1].date() != _t.date():
                for _ax_sep in [ax, ax_kline, ax_stoch_1h,
                                  ax_stoch_15m, ax_sq_main]:
                    _ax_sep.axvline(_i - 0.5, color="gray",
                                     linestyle=":", alpha=0.5,
                                     linewidth=0.8, zorder=1)

    # v3.7.25: st.pyplot(fig) 推迟到 5 子图全部绘制完毕之后

    # ── Stoch RSI 助手 (3 处面板共用) ──
    def _stoch_rsi(close: pd.Series, period: int = 14):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=3).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=3).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rlow = rsi.rolling(period, min_periods=3).min()
        rhigh = rsi.rolling(period, min_periods=3).max()
        k_raw = ((rsi - rlow) / (rhigh - rlow).replace(0, np.nan)) * 100
        k = k_raw.rolling(3, min_periods=1).mean()
        d = k.rolling(3, min_periods=1).mean()
        return k, d

    def _zone_label(k_val, d_val):
        if k_val is None or pd.isna(k_val):
            return "—", "#9E9E9E"
        if k_val >= 80:
            return "超买", "#E53935"
        if k_val <= 20:
            return "超卖", "#43A047"
        if k_val > d_val and k_val < 50:
            return "超卖反转↑", "#1E88E5"
        if k_val < d_val and k_val > 50:
            return "超买回落↓", "#FB8C00"
        return "中性", "#757575"

    def _last_pair(k, d):
        if k is None:
            return None, None
        kk = k.dropna()
        dd = d.dropna()
        if len(kk) == 0 or len(dd) == 0:
            return None, None
        return float(kk.iloc[-1]), float(dd.iloc[-1])

    # ── 主图下方: 1h Stoch RSI + 15m Stoch RSI (盘中实时, v3.7.23) ──
    @st.cache_data(ttl=300)
    def _fetch_kline_for_stoch(ticker, interval, period):
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
        return None

    _futures_t_top = "GC=F" if asset_key == "GLD" else "SI=F"
    # 共享时间窗口: 1h 和 15m 都用 _intraday_days
    # yfinance 15m 数据上限 60 天, 30 天足够覆盖默认 14 天 + 边距
    _stoch_window = _intraday_days if _is_1h_view else 14
    _stoch_window = min(_stoch_window, 30)  # 不超过 yfinance 15m 上限
    _kline_1h_top = _fetch_kline_for_stoch(_futures_t_top, "1h",
                                             f"{max(_stoch_window+5, 30)}d")
    _kline_15m_top = _fetch_kline_for_stoch(_futures_t_top, "15m",
                                              f"{_stoch_window+3}d")

    # 把 1h/15m Stoch RSI 投影到主图整数 x (plot_ts 索引)
    # 这样跟主图、K 线、Squeeze 全部 sharex 对齐
    if _is_1h_view and len(plot_dates) > 0:
        # plot_dates 是 1h 时间戳列表 (整数 x = 索引)
        _ref_nums_top = np.array([mdates.date2num(t) for t in plot_dates])
        _n_ref_top = len(plot_dates)

        def _proj_to_main_idx(ts_index):
            """把 datetime 索引投影到主图整数 x."""
            ts_nums = np.array([mdates.date2num(t) for t in ts_index])
            return np.interp(ts_nums, _ref_nums_top, np.arange(_n_ref_top))

        def _draw_stoch_on_main(target_ax, kline, label):
            """在 target_ax (sharex 子图) 上绘制 Stoch RSI, 用主图整数 x."""
            if kline is None or len(kline) < 30:
                target_ax.text(0.5, 0.5, f"{label} 数据暂时不可用",
                                 transform=target_ax.transAxes,
                                 ha="center", va="center", color="gray")
                return
            k_full, d_full = _stoch_rsi(kline["Close"])
            t_start = plot_dates[0]
            t_end = plot_dates[-1]
            mask = (kline.index >= t_start) & (kline.index <= t_end)
            idx = kline.index[mask]
            if len(idx) == 0:
                return
            x_vals = _proj_to_main_idx(idx)
            target_ax.plot(x_vals, k_full[mask].values,
                            color="#1E88E5", lw=1.2, label="K")
            target_ax.plot(x_vals, d_full[mask].values,
                            color="#FB8C00", lw=1.0, label="D")
            target_ax.axhline(80, color="#E53935", ls="--", lw=0.5, alpha=0.5)
            target_ax.axhline(20, color="#43A047", ls="--", lw=0.5, alpha=0.5)
            target_ax.axhspan(0, 20, alpha=0.05, color="green")
            target_ax.axhspan(80, 100, alpha=0.05, color="red")
            target_ax.set_ylim(-2, 102)
            # 当前值放在右上角 (避免用 title 把图分开)
            k_clean = k_full[mask].dropna()
            d_clean = d_full[mask].dropna()
            if len(k_clean) > 0 and len(d_clean) > 0:
                last_k = float(k_clean.iloc[-1])
                last_d = float(d_clean.iloc[-1])
                zone, color = _zone_label(last_k, last_d)
                target_ax.text(0.99, 0.92,
                               f"{label}: K={last_k:.0f} D={last_d:.0f} ({zone})",
                               transform=target_ax.transAxes,
                               ha="right", va="top", fontsize=9,
                               color=color, fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.3",
                                         facecolor="white", alpha=0.85,
                                         edgecolor=color))
            target_ax.set_ylabel(label, fontsize=9)
            target_ax.legend(loc="upper left", fontsize=7)
            target_ax.grid(alpha=0.3)

        _draw_stoch_on_main(ax_stoch_1h, _kline_1h_top, "1h Stoch")
        _draw_stoch_on_main(ax_stoch_15m, _kline_15m_top, "15m Stoch")
    else:
        # 日线模式: 隐藏 Stoch / K线 / Squeeze 子图
        for _ax_hide in [ax_stoch_1h, ax_stoch_15m, ax_kline, ax_sq_main]:
            _ax_hide.set_visible(False)

    # ── 盘中 K线 + Squeeze (v3.7.25 合并到主图 sharex 子图) ──
    # 不再独立 fig2, 用 ax_kline + ax_sq_main 子图共用主图整数 x
    @st.cache_data(ttl=300)
    def _fetch_futures_kline(ticker, interval, period="5d"):
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
        return None

    _futures_ticker = "GC=F" if asset_key == "GLD" else "SI=F"
    _futures_name = "COMEX Gold" if asset_key == "GLD" else "COMEX Silver"
    # 强制用 1h 与主图对齐, 数据源 = gld_1h.csv (与主图 _1h 同步)
    # v3.7.80: 用 user-selected interval (默认 15m)
    _kline_interval = locals().get("_interval_code", "1h") if _is_1h_view else "1h"
    _kline_data = gld_1h if gld_1h is not None else _fetch_futures_kline(_futures_ticker, "1h")
    _kline_label = f"{_futures_name} {_kline_interval}"  # v3.7.85: 跟实际 interval

    if _kline_data is not None and len(_kline_data) > 0 and _is_1h_view:
        # v3.7.25: 强制与主图同窗口 (plot_dates), 共享整数 x
        # _1h 用主图 _1h (已加载, 是 gld_1h windowed)
        # full 数据用更多 warmup 算指标
        _warmup_extra = timedelta(days=10)  # 算 BB/Keltner 需要 60 bars warmup
        _kl_full_mask = _kline_data.index >= (plot_dates[0] - _warmup_extra)
        _1h_full = _kline_data[_kl_full_mask].copy()
        # v3.7.78: dirty bar 直接 drop (而非 clip 到边界, 避免假低谷)
        # 判定: Close 偏离 daily Low/High 超 1.5% = bar 整体异常 (yfinance prepost 偶发)
        # v3.7.84: 期货源跳过 — overnight legitimately 超 ETF daily L/H
        def _drop_dirty(df):
            if _kline_is_futures:
                return df
            d_norms = df.index.normalize()
            d_lo = pd.Series(low_d.reindex(d_norms.unique()),
                              index=low_d.reindex(d_norms.unique()).index)
            d_hi = pd.Series(high_d.reindex(d_norms.unique()),
                              index=high_d.reindex(d_norms.unique()).index)
            lo_map = df.index.to_series().apply(
                lambda t: d_lo.get(t.normalize(), float("nan")))
            hi_map = df.index.to_series().apply(
                lambda t: d_hi.get(t.normalize(), float("nan")))
            df = df.copy()
            close_dirty = (((df["Close"] < lo_map * 0.985)
                              | (df["Close"] > hi_map * 1.015))
                            & lo_map.notna())
            df = df[~close_dirty]
            df["Low"] = df["Low"].clip(lower=lo_map * 0.997)
            df["High"] = df["High"].clip(upper=hi_map * 1.003)
            return df
        _1h_full = _drop_dirty(_1h_full)
        _c1h_full = _1h_full["Close"]
        _h1h_full = _1h_full["High"]
        _l1h_full = _1h_full["Low"]
        _o1h_full = _1h_full["Open"]

        # 显示范围 = 主图相同 plot_dates, 同样 drop dirty
        _1h = _kline_data.reindex(plot_dates)
        _1h = _drop_dirty(_1h)
        _c1h, _h1h, _l1h, _o1h = _1h["Close"], _1h["High"], _1h["Low"], _1h["Open"]

        # ── 用 full 数据计算指标, 然后截取显示范围 ──

        # Stoch RSI (14, 14, 3, 3) — 使用同一个 _stoch_rsi 助手, 与上方 MTF 面板一致
        _stoch_rsi_k, _stoch_rsi_d = _stoch_rsi(_c1h_full)

        # Squeeze: BB vs Keltner
        _bb_len = 20
        _kc_len = 20
        _kc_mult = 1.5
        _sma = _c1h_full.rolling(_bb_len, min_periods=5).mean()
        _std = _c1h_full.rolling(_bb_len, min_periods=5).std()
        _bb_upper = _sma + 2 * _std
        _bb_lower = _sma - 2 * _std

        _tr = pd.concat([_h1h_full - _l1h_full,
                          (_h1h_full - _c1h_full.shift(1)).abs(),
                          (_l1h_full - _c1h_full.shift(1)).abs()], axis=1).max(axis=1)
        _atr = _tr.rolling(_kc_len, min_periods=5).mean()
        _kc_upper = _sma + _kc_mult * _atr
        _kc_lower = _sma - _kc_mult * _atr

        _squeeze_on = (_bb_upper < _kc_upper) & (_bb_lower > _kc_lower)
        _squeeze_off = ~_squeeze_on
        _mom = _c1h_full - _sma

        # 截取到显示范围
        _stoch_rsi_k = _stoch_rsi_k.reindex(_1h.index)
        _stoch_rsi_d = _stoch_rsi_d.reindex(_1h.index)
        _bb_upper = _bb_upper.reindex(_1h.index)
        _bb_lower = _bb_lower.reindex(_1h.index)
        _kc_upper = _kc_upper.reindex(_1h.index)
        _kc_lower = _kc_lower.reindex(_1h.index)
        _squeeze_on = _squeeze_on.reindex(_1h.index).fillna(False)
        _squeeze_off = _squeeze_off.reindex(_1h.index).fillna(True)
        _mom = _mom.reindex(_1h.index)

        # index-based x
        _idx_1h = list(_1h.index)
        _d2i_1h = {d: i for i, d in enumerate(_idx_1h)}
        def _xi1h(d): return _d2i_1h.get(d)
        def _xi1h_arr(dates): return [_d2i_1h[d] for d in dates if d in _d2i_1h]

        # v3.7.36: 与主图统一时间格式 (sharex 下 bottom 子图 formatter 覆盖主图)
        # 天数 ≤ 5: 显示日期+时间; > 5: 仅日期 (避免 24h 时间充斥)
        def _fmt1h(x, pos):
            idx = int(round(x))
            if 0 <= idx < len(_idx_1h):
                dt = _idx_1h[idx]
                if _show_hours:
                    # 短窗口: 日变化时显示日期+时间, 否则仅时间
                    if idx == 0 or dt.date() != _idx_1h[idx - 1].date():
                        return dt.strftime("%m/%d\n%H:%M")
                    return dt.strftime("%H:%M")
                else:
                    # 长窗口: 仅日变化时显示日期, 其余空白
                    if idx == 0 or dt.date() != _idx_1h[idx - 1].date():
                        return dt.strftime("%m/%d")
                    return ""
            return ""

        # v3.7.25: 复用合并 fig 的 ax_kline / ax_sq_main, 不再独立 fig2
        ax_price = ax_kline
        ax_sq = ax_sq_main

        # 真实 K线 (红绿蜡烛图)
        _body_w = 0.6
        _wick_w = 0.15
        # v3.7.64: candle 数据 (ETF 价) × _r 转期货坐标 (跟 y 轴一致)
        for dt in _1h.index:
            ix = _xi1h(dt)
            if ix is None:
                continue
            o_raw, h_raw, l_raw, c_raw = (_o1h.get(dt, 0), _h1h.get(dt, 0),
                                            _l1h.get(dt, 0), _c1h.get(dt, 0))
            if o_raw == 0 or c_raw == 0:
                continue
            o, h, l, c = o_raw * _r, h_raw * _r, l_raw * _r, c_raw * _r
            color = "#4CAF50" if c >= o else "#F44336"
            # 影线
            ax_price.plot([ix, ix], [l, h], color=color, lw=_wick_w * 2, zorder=2)
            # 实体
            body_bottom = min(o, c)
            body_height = abs(c - o) if abs(c - o) > 0.5 else 0.5 * _r
            ax_price.bar(ix, body_height, bottom=body_bottom, width=_body_w,
                         color=color, edgecolor=color, zorder=3)
        # v3.7.64: BB/Keltner 也 × _r 转期货坐标 (与 candle 一致)
        _bb_u_clean = _bb_upper.dropna()
        _bb_l_clean = _bb_lower.dropna()
        if len(_bb_u_clean) > 0:
            ax_price.plot(_xi1h_arr(_bb_u_clean.index),
                          _bb_u_clean.values * _r,
                          color="blue", lw=0.6, alpha=0.4)
            ax_price.plot(_xi1h_arr(_bb_l_clean.index),
                          _bb_l_clean.values * _r,
                          color="blue", lw=0.6, alpha=0.4)
        _kc_u_clean = _kc_upper.dropna()
        _kc_l_clean = _kc_lower.dropna()
        if len(_kc_u_clean) > 0:
            ax_price.plot(_xi1h_arr(_kc_u_clean.index),
                          _kc_u_clean.values * _r,
                          color="orange", lw=0.6, ls="--", alpha=0.4)
            ax_price.plot(_xi1h_arr(_kc_l_clean.index),
                          _kc_l_clean.values * _r,
                          color="orange", lw=0.6, ls="--", alpha=0.4)

        # Squeeze 背景色
        for i, dt in enumerate(_idx_1h):
            if _squeeze_on.get(dt, False):
                ax_price.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="red")

        # v3.7.53: 数据过期时, 入场窗口高亮关掉
        _ld_zone = pd.Timestamp(last_date)
        _stale_zone = (pd.Timestamp(today_sgt) - _ld_zone.normalize()).days > 1

        # 入场窗口标注: 当日线有买入信号时, Stoch RSI < 30 的区域高亮
        _has_buy_signal = False
        _signal_type_today = ""
        if not _stale_zone and last_date in _unified_viz_raw.index:
            _chosen_today = _unified_viz_raw.loc[last_date, "chosen"]
            if _chosen_today in ("BUY CALL", "SELL PUT"):
                _has_buy_signal = True
                _signal_type_today = _chosen_today
        # 也检查最近2天 — 但若数据过期就不检查
        if not _stale_zone:
            for _dd in _unified_viz_raw.index[-3:]:
                _ch = _unified_viz_raw.loc[_dd, "chosen"]
                if _ch in ("BUY CALL", "SELL PUT"):
                    _has_buy_signal = True
                    _signal_type_today = _ch

        # "反转穿越20 + 近BB下轨" = 最优入场信号 (61%胜率 on 1h)
        _cross_20_up = (_stoch_rsi_k > 20) & (_stoch_rsi_k.shift(1) <= 20)
        _at_bb_low = _c1h <= _bb_lower.reindex(_c1h.index) * 1.002
        _entry_window = _cross_20_up & _at_bb_low  # 最优组合
        _entry_zone = _at_bb_low & (_stoch_rsi_k < 30)  # 准备区

        if _has_buy_signal:
            # 入场窗口 (绿色): 反转穿越20 + 近BB下轨
            for i, dt in enumerate(_idx_1h):
                if _entry_window.get(dt, False):
                    ax_price.axvspan(i - 0.5, i + 0.5, alpha=0.25,
                                      color="#4CAF50", zorder=1)
                elif _entry_zone.get(dt, False):
                    # 准备区 (浅绿): 接近但未确认
                    ax_price.axvspan(i - 0.5, i + 0.5, alpha=0.08,
                                      color="#4CAF50", zorder=1)

        # v3.7.85: 标题信息合并到 ylabel, 主图和副图间不插独立 title
        _last_price = _c1h.iloc[-1] * _r
        ax_price.set_ylabel(
            f"{_kline_label} ${_last_price:.1f} "
            f"({_1h.index[-1].strftime('%m/%d %H:%M')} ET)",
            fontsize=9)
        ax_price.grid(True, alpha=0.3)
        # v3.7.80: K 线 chart 图例 — 期权/期货 marker 区别 (Line2D 已 module-import)
        _kline_legend = [
            Line2D([0],[0], marker="^", color="w", markerfacecolor="#2196F3",
                    markeredgecolor="black", markersize=10,
                    label="▲ BUY CALL 期权 (盘中)"),
            Line2D([0],[0], marker="^", color="w", markerfacecolor="#FF9800",
                    markeredgecolor="black", markersize=10,
                    label="▲ SELL PUT 期权 (盘中)"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#2196F3",
                    markeredgecolor="black", markersize=10,
                    label="◇ BUY 期货 (非盘中, 24h)"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#FF9800",
                    markeredgecolor="black", markersize=10,
                    label="◇ SELL 期货 (非盘中)"),
            Line2D([0],[0], marker="v", color="w", markerfacecolor="#F44336",
                    markeredgecolor="black", markersize=10,
                    label="▼ EXIT"),
        ]
        ax_price.legend(handles=_kline_legend, loc="upper left",
                        fontsize=7, framealpha=0.85, ncol=3)

        # ── 把 1h/15m 时间戳投影到 K线索引 x (与 price/squeeze 对齐) ──
        _ref_nums = np.array([mdates.date2num(t) for t in _idx_1h])
        _n_ref = len(_idx_1h)

        def _proj_to_idx(ts_index):
            """把 datetime 索引投影到 [0, n_ref-1] 浮点 x."""
            ts_nums = np.array([mdates.date2num(t) for t in ts_index])
            return np.interp(ts_nums, _ref_nums, np.arange(_n_ref))

        # ── 盘中触发: 检测显示窗口内所有触发, 持久化, 画散点 ──
        from core.intraday_triggers import (
            detect_triggers as _ig_detect,
            TriggerConfig as _IG_Cfg,
            DEFAULT_BUY_RULES as _IG_BUY,
            DEFAULT_EXIT_RULES as _IG_EXIT,
            upsert_log as _ig_upsert,
            worst_of_day as _ig_worst,
            dedupe_intraday as _ig_dedupe,  # v3.7.67
        )
        _interval_min = {"1h": 60, "30m": 30, "15m": 15, "5m": 5}.get(
            _kline_interval, 60)
        _log_path_intra = os.path.join(_intra_cfg["data_root"],
                                        "intraday_signal_log.parquet")
        _thresholds_intra = sig_df[["bp030_price", "bp090_price"]]

        _live_buys_raw = _ig_detect(
            _kline_data, _thresholds_intra,
            _IG_Cfg(timeframe_minutes=_interval_min, side="BUY",
                    rule_set=_IG_BUY, confirm_mode="all"),  # v3.7.83
            asset=asset_key, daily_low=low_d, daily_high=high_d)
        _live_exits_raw = _ig_detect(
            _kline_data, _thresholds_intra,
            _IG_Cfg(timeframe_minutes=_interval_min, side="EXIT",
                    rule_set=_IG_EXIT, confirm_mode="all"),  # v3.7.83
            asset=asset_key, daily_low=low_d, daily_high=high_d)
        # v3.7.67: 日内去重 — 同日多触发只保留显著加仓点 (≥0.5% 跌幅)
        _live_buys = _ig_dedupe(_live_buys_raw, side="BUY", min_drop_pct=0.3)
        _live_exits = _ig_dedupe(_live_exits_raw, side="EXIT", min_drop_pct=0.3)

        # 写入持久 log (去重交给 upsert)
        try:
            if len(_live_buys) > 0:
                _ig_upsert(_live_buys, _log_path_intra)
            if len(_live_exits) > 0:
                _ig_upsert(_live_exits, _log_path_intra)
        except Exception as _e_log:
            st.caption(f"日志写入失败: {_e_log}")

        # 截到显示窗口画散点
        _w_start, _w_end = _idx_1h[0], _idx_1h[-1]

        # v3.7.74: 直接从 intraday_signal_log 读触发 + 当场 dedupe (而非依赖
        # _live_buys, 后者有时丢数据). log 已确认含 4-29 09:30 EDT \$414.16.
        from core.intraday_triggers import dedupe_intraday as _dd_chart
        if len(_intra_log_asset) > 0:
            # v3.7.80: 按 timeframe 过滤 — 只显示当前 interval 触发, 避免 1h+15m 混合
            _tf_match = f"{_interval_code.replace('m','')}m" \
                if _interval_code != "1h" else "60m"
            _log_buys = _intra_log_asset[
                (_intra_log_asset["side"] == "BUY") &
                (_intra_log_asset.get("timeframe", "") == _tf_match)]
            _log_exits = _intra_log_asset[
                (_intra_log_asset["side"] == "EXIT") &
                (_intra_log_asset.get("timeframe", "") == _tf_match)]
            # dedupe 每天独立, 用 daily Low 做 sanity 截 dirty 价位
            def _dedupe_per_day(log_df, side):
                if not len(log_df):
                    return log_df
                rows = []
                for d, grp in log_df.groupby("date"):
                    # 截 dirty trigger price (1h log 可能含 prepost 异常如 \$401)
                    d_norm = pd.Timestamp(d).normalize()
                    d_lo = float(low_d.get(d_norm, 0)) if d_norm in low_d.index else 0
                    d_hi = float(high_d.get(d_norm, 0)) if d_norm in high_d.index else 1e9
                    grp_clean = grp.copy()
                    if d_lo > 0:
                        grp_clean["price"] = grp_clean["price"].clip(
                            lower=d_lo * 0.997)
                    if d_hi > 0:
                        grp_clean["price"] = grp_clean["price"].clip(
                            upper=d_hi * 1.003)
                    dd = _dd_chart(grp_clean, side=side, min_drop_pct=0.3)
                    if len(dd):
                        rows.append(dd)
                if not rows:
                    return log_df.iloc[0:0]
                return pd.concat(rows, ignore_index=True)
            _live_buys = _dedupe_per_day(_log_buys, "BUY")
            _live_exits = _dedupe_per_day(_log_exits, "EXIT")

        # debug expander — 让用户验证 dedupe 数据
        if len(_live_buys) > 0:
            _dbg = _live_buys[
                (_live_buys["trigger_time"] >= _w_start) &
                (_live_buys["trigger_time"] <= _w_end)].copy()
            with st.expander(f"🔍 1h chart BUY triggers ({len(_dbg)} 笔, dedupe 后)",
                              expanded=False):
                if len(_dbg):
                    _dbg_show = _dbg[["trigger_time","price","rules"]].copy()
                    _dbg_show["futures_price"] = (
                        _dbg_show["price"] * _r).round(2)
                    st.dataframe(_dbg_show, use_container_width=True)
                else:
                    st.caption("窗口内无触发")
        # v3.7.70: 1h marker 用跟主图同样的图例 (BUY CALL/SELL PUT/EXIT 颜色)
        # 颜色根据当日 chosen 决定 (BC=蓝 ▲ / SP=橙 ▲ / EXIT=红 ▼)
        # 如有 STRADDLE/SHORT_VOL 同时触发, 单独画 ★/P
        _SIG_PAL = {"BUY CALL": ("#2196F3", "^"), "SELL PUT": ("#FF9800", "^"),
                     "EXIT": ("#F44336", "v"), "STRADDLE": ("#FFD700", "*"),
                     "SHORT_VOL": ("#FF6F00", "P")}

        def _trig_palette(trig_row, side):
            # 用当日 sig_df 的 buy_type 决定 BUY 颜色
            d = pd.Timestamp(trig_row["trigger_time"]).normalize()
            if side == "BUY" and d in sig_df.index:
                bt = sig_df.loc[d].get("buy_type")
                if bt in ("BUY CALL", "SELL PUT"):
                    return _SIG_PAL[bt]
                return _SIG_PAL["BUY CALL"]  # default
            if side == "EXIT":
                return _SIG_PAL["EXIT"]
            return ("#888", "o")

        for _trigs, _side in [(_live_buys, "BUY"), (_live_exits, "EXIT")]:
            if len(_trigs) == 0:
                continue
            _disp = _trigs[(_trigs["trigger_time"] >= _w_start) &
                           (_trigs["trigger_time"] <= _w_end)]
            if len(_disp) == 0:
                continue
            # v3.7.77: 期权 vs 期货 marker 形状区分
            # 期权 (盘中 9:30-16 ET): 实心 ▲/v (BUY/EXIT)
            # 期货 (非盘中): ◇ 菱形 (BUY) / ◆ 实心菱形 + 灰色 (期货 EXIT)
            #   形状不同避免跟期权 ▲ 混淆 (颜色仍按 BC/SP 区分)
            for _, _r2 in _disp.iterrows():
                _t = pd.Timestamp(_r2["trigger_time"])
                _xx = _proj_to_idx([_t])[0]
                _color, _marker_opt = _trig_palette(
                    {"trigger_time": _t}, _side)
                _h = _t.hour + _t.minute / 60.0
                _in_us = 9.5 <= _h < 16.0
                _y = _r2["price"] * _r
                if _in_us:
                    # 期权: 实心 ▲/v + 黑边
                    ax_price.scatter([_xx], [_y],
                                     marker=_marker_opt, s=180, color=_color,
                                     edgecolors="black", lw=1.2, zorder=10)
                    _label_tag = ""
                else:
                    # 期货: ◇ 菱形, 颜色仍按 BC/SP, 加 "F" 文字标
                    _fut_marker = "D" if _side == "BUY" else "d"  # D 实心菱形
                    ax_price.scatter([_xx], [_y],
                                     marker=_fut_marker, s=180, color=_color,
                                     edgecolors="black", lw=1.2, zorder=10)
                    _label_tag = " (期货)"
                # 价位 + 时间 + 工具类型 标注
                ax_price.annotate(
                    f"${_y:.0f}{_label_tag}",
                    xy=(_xx, _y),
                    xytext=(8, -3),
                    textcoords="offset points",
                    fontsize=7, color=_color,
                    fontweight="bold")

        # (v3.7.23: Stoch RSI 子图已移到主图下方, 此 K线 panel 仅保留 K线 + Squeeze)

        # Squeeze Momentum
        _mom_clean = _mom.dropna()
        if len(_mom_clean) > 0:
            xi_mom = _xi1h_arr(_mom_clean.index)
            colors_mom = ["#4CAF50" if v >= 0 else "#F44336" for v in _mom_clean.values]
            # 颜色深浅: 增加中 vs 减弱中
            for i in range(len(xi_mom)):
                v = _mom_clean.values[i]
                if i > 0:
                    prev = _mom_clean.values[i - 1]
                    if v >= 0:
                        c_sq = "#4CAF50" if v > prev else "#81C784"
                    else:
                        c_sq = "#F44336" if v < prev else "#EF9A9A"
                else:
                    c_sq = "#4CAF50" if v >= 0 else "#F44336"
                ax_sq.bar(xi_mom[i], v, width=0.8, color=c_sq, edgecolor="none")

        # Squeeze on/off 标记
        for i, dt in enumerate(_idx_1h):
            if _squeeze_on.get(dt, False):
                ax_sq.scatter([i], [0], marker="o", s=10, color="red", zorder=5)
            elif _squeeze_off.get(dt, False) and i > 0 and _squeeze_on.get(_idx_1h[i-1], False):
                ax_sq.scatter([i], [0], marker="o", s=15, color="green", zorder=5)

        ax_sq.axhline(0, color="black", lw=0.5)
        ax_sq.set_ylabel("Squeeze Mom")
        ax_sq.grid(True, alpha=0.3)
        # v3.7.36: 长窗口 (>5d) 改用日界 ticks, 短窗口用均匀 ticks
        ax_sq.xaxis.set_major_formatter(FuncFormatter(_fmt1h))
        if _show_hours:
            ax_sq.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
        else:
            # 长窗口: 把 tick 放在每日第一根 1h bar 上
            day_change_idx = [i for i, ts in enumerate(_idx_1h)
                                if i == 0 or ts.date() != _idx_1h[i-1].date()]
            # 每隔 N 天 (避免太密)
            step = max(1, len(day_change_idx) // 12)
            ax_sq.set_xticks(day_change_idx[::step])

        # 全部 5 子图绘制完毕, 一次性渲染合并 fig
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # 当前状态文字
        _last_1h = _1h.index[-1]
        _srk = _stoch_rsi_k.get(_last_1h, 50)
        _srd = _stoch_rsi_d.get(_last_1h, 50)
        _sq_on = _squeeze_on.get(_last_1h, False)
        _mom_v = _mom.get(_last_1h, 0)

        zone_1h = "超卖" if _srk < 20 else ("超买" if _srk > 80 else "中性")
        sq_state = "挤压中(蓄力)" if _sq_on else "已释放"
        mom_dir = "向上" if _mom_v > 0 else "向下"

        st.markdown(f"**当前 ({_last_1h.strftime('%m/%d %H:%M')})**: "
                    f"Stoch RSI K={_srk:.0f} D={_srd:.0f} ({zone_1h}) | "
                    f"Squeeze: {sq_state} | 动量: {mom_dir}")

        _is_entry_now = _entry_window.get(_last_1h, False)
        _is_entry_zone = _entry_zone.get(_last_1h, False)

        if _has_buy_signal and _is_entry_now:
            st.success(f"**{_signal_type_today} + 反转确认 + BB下轨 → 入场!** "
                       f"(K={_srk:.0f}, 61%历史胜率)")
        elif _has_buy_signal and _is_entry_zone:
            st.success(f"**{_signal_type_today} + 接近BB下轨 → 准备入场, 等反转确认** "
                       f"(K={_srk:.0f})")
        elif _has_buy_signal and _srk > 80:
            st.warning(f"**持仓中 + 超买 → 考虑止盈** (K={_srk:.0f})")
        elif _has_buy_signal:
            st.info(f"{_signal_type_today} 活跃 — 等待价格接近BB下轨 + Stoch RSI反转 (K={_srk:.0f})")
        elif _is_entry_zone:
            st.info(f"接近BB下轨 + 超卖 — 等待日线买入信号确认 (K={_srk:.0f})")
        elif _srk > 80:
            st.warning("超买区 — 如有持仓注意止盈")
        elif _sq_on:
            st.info("Squeeze挤压中 → 波动率压缩, 等待突破")
    else:
        # 数据缺失或日线模式: 仍渲染 fig (主图) 但跳过 K 线 panel
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if _kline_data is None or len(_kline_data) == 0:
            st.caption("GC=F K线数据暂时不可用")

    # ── 期权策略实时面板 (4 策略并列, 当日活跃信号高亮) ──
    st.divider()
    st.subheader("期权策略实时面板")
    st.caption("4 类策略并列展示, 入场信号触发时对应策略高亮 ✅, 否则灰显示 (未激活)")

    # v3.7.53/54: 数据过期 OR 当前 bp_est ≥ 0.30 时, 屏蔽方向性"激活"
    # (今日已触过但已反弹的情况不应继续显"激活")
    _ld2 = pd.Timestamp(last_date)
    _stale_today = (pd.Timestamp(today_sgt) - _ld2.normalize()).days > 1
    _live_actionable = bp_est is not None and bp_est < 0.30

    # 当日各策略激活状态
    _r2 = sig_df.loc[last_date] if (last_date in sig_df.index and not _stale_today) else None
    _uni_today = (_unified_viz_raw.loc[last_date]
                  if (last_date in _unified_viz_raw.index and not _stale_today)
                  else None)
    _chosen_today = _uni_today["chosen"] if _uni_today is not None else None
    # 方向性 (BC/SP) 必须实时 bp 仍在 buy zone 才算激活
    _is_buy_call = (_chosen_today is not None
                     and "BUY CALL" in _chosen_today and _live_actionable)
    _is_sell_put = (_chosen_today is not None
                     and "SELL PUT" in _chosen_today and _live_actionable)
    # 波动率信号不依赖 bp (是 vol 触发, 不是 band)
    _is_straddle_now = (_chosen_today is not None and "STRADDLE" in _chosen_today)
    _is_short_vol_now = (_chosen_today is not None and "SHORT_VOL" in _chosen_today)
    if _stale_today:
        st.warning(f"⚠️ 数据过期 (末日 {_ld2.date()}, 距今 "
                   f"{(pd.Timestamp(today_sgt)-_ld2.normalize()).days}d) — "
                   f"4 策略实时激活均强制视为未触发. 重启 dashboard 触发刷新.")
    elif _chosen_today and not _live_actionable and _chosen_today in ("BUY CALL","SELL PUT"):
        st.info(f"ℹ️ 方向性信号 {_chosen_today} 今日触过但已反弹 (现 bp={bp_est:.2f}), "
                f"实时面板 BC/SP 不再激活. 历史触发见可视化 marker.")

    # 当前价位 + 估算 1σ (5d hold)
    _spot = gc_now if gc_now > 0 else last_close * _viz_ratio
    _gld_spot = last_close
    _sigma_5d = (rv / 100) * (5/252)**0.5 * _gld_spot  # GLD ATM 1σ ($)
    _sigma_pct = (rv / 100) * (5/252)**0.5 * 100       # 1σ %

    # 推荐 DTE 范围 (期权 hold_days=5 但 DTE 选 21-45 sweet spot)
    _dte_buy = "30-45 DTE"   # Long Call/Put 长期权选择
    _dte_sell = "21-30 DTE"  # Short Put / IC 短期权选择 (theta 加速段)

    cols_strat = st.columns(4)

    # 策略 1: BUY CALL (低 RV, 期货优先 / 期权备选)
    with cols_strat[0]:
        if _is_buy_call:
            st.success(f"✅ **BUY CALL 激活**\n\n"
                       f"**首选: 期货多头 + 3% 止损**\n"
                       f"(实证 96% wr, 见 v3.6.6)\n\n"
                       f"备选: Long Call\n"
                       f"- DTE: {_dte_buy}\n"
                       f"- Strike: ATM (~${_gld_spot:.0f})\n"
                       f"- 最小成本 ≈ {_sigma_pct:.1f}% × spot")
        else:
            st.markdown(f"⚪ **BUY CALL** (未激活)\n\n"
                        f"触发条件: Bull + bp_low<0.30 + RV%tile<0.50\n\n"
                        f"备选工具: 期货 / Long Call\n"
                        f"DTE: {_dte_buy}")

    # 策略 2: SELL PUT (高 RV, 期权 100% 胜率, 期货不优)
    with cols_strat[1]:
        if _is_sell_put:
            _put_strike_low = _gld_spot - 1.0 * _sigma_5d
            st.success(f"✅ **SELL PUT 激活**\n\n"
                       f"**首选: 期权 Sell Put**\n"
                       f"(实证 100% wr, 见 v3.6.6)\n\n"
                       f"- DTE: {_dte_sell}\n"
                       f"- Strike: 1σ 下方 ≈ ${_put_strike_low:.0f}\n"
                       f"- 收 premium ≈ {_sigma_pct*0.5:.2f}%\n"
                       f"- ⚠️ 期货 wr 仅 68%, 不推荐")
        else:
            st.markdown(f"⚪ **SELL PUT** (未激活)\n\n"
                        f"触发条件: Bull + bp_low<0.30 + RV%tile>0.85\n\n"
                        f"工具: 期权 Sell Put (高 RV 收 IV)\n"
                        f"DTE: {_dte_sell}")

    # 策略 3: Long Straddle (做多波动率)
    with cols_strat[2]:
        if _is_straddle_now:
            _call_strike = round(_gld_spot)
            _put_strike = round(_gld_spot)
            # IV/RV 比率
            _gvz = features['gvz'].get(last_date, np.nan) \
                if 'gvz' in features.columns else np.nan
            _iv_rv = _gvz / rv if (not np.isnan(_gvz) and rv > 0) else np.nan
            _iv_warn = (f"\n📊 IV/RV={_iv_rv:.2f}"
                         if not np.isnan(_iv_rv) else "")
            st.success(f"✅ **Long Straddle 激活**\n\n"
                       f"- DTE: {_dte_buy}\n"
                       f"- Strike: ATM ${_call_strike}\n"
                       f"  Long Call ${_call_strike} +\n"
                       f"  Long Put ${_put_strike}\n"
                       f"- 成本 ≈ {_sigma_pct*2:.1f}% (双腿)\n"
                       f"- 赢条件: |move| > 1σ ({_sigma_pct:.1f}%)\n"
                       f"- 50% 利润早平"
                       + _iv_warn)
        else:
            st.markdown(f"⚪ **Long Straddle** (未激活)\n\n"
                        f"触发: RV<20% + 临 FOMC/NFP/OPEX, score≥3\n\n"
                        f"DTE: {_dte_buy} (sweet spot, gamma+vega 均衡)\n"
                        f"非 5-DTE (gamma 太极端)")

    # 策略 4: Iron Condor (做空波动率)
    with cols_strat[3]:
        if _is_short_vol_now:
            _ic_short_call = round(_gld_spot + 1.6 * _sigma_5d)
            _ic_long_call = round(_gld_spot + 3.0 * _sigma_5d)
            _ic_short_put = round(_gld_spot - 1.6 * _sigma_5d)
            _ic_long_put = round(_gld_spot - 3.0 * _sigma_5d)
            st.success(f"✅ **Iron Condor 激活**\n\n"
                       f"- DTE: {_dte_sell}\n"
                       f"- Short Put: ${_ic_short_put} (1.6σ)\n"
                       f"- Long Put:  ${_ic_long_put} (3σ 翼)\n"
                       f"- Short Call: ${_ic_short_call} (1.6σ)\n"
                       f"- Long Call:  ${_ic_long_call} (3σ 翼)\n"
                       f"- 收 credit ≈ {_sigma_pct*0.4:.2f}%\n"
                       f"- 50% credit 早平")
        else:
            st.markdown(f"⚪ **Iron Condor** (未激活)\n\n"
                        f"触发: RV%tile∈[0.35,0.65] + 远离事件 + 趋势回落\n\n"
                        f"DTE: {_dte_sell}\n"
                        f"非 5-DTE (gamma 风险)")

    # ── DTE 与 持仓天数说明 ──
    with st.expander("ℹ️ DTE vs 持仓天数 (展开看)"):
        st.markdown("""
        - **DTE (Days To Expiry)**: 期权到期日距今天数 — 决定 theta 衰减速度
        - **持仓天数 (Holding Period)**: 系统 hold_days=5d 是**平均持仓**, 不是 DTE
        - 实战流程: 选 30 DTE 期权链 → 持仓 5 天 (变成 25 DTE) → 50% credit 早平

        | 策略 | 推荐 DTE | 原因 |
        |------|---------|------|
        | Long Call/Put / Straddle | 30-45 DTE | gamma 适中, vega 充足 |
        | Sell Put / Iron Condor | 21-30 DTE | theta 加速段 (sweet spot) |

        **不用 5-DTE 期权**: gamma 极端, 价格小动就被 ITM, gamma 风险 >> theta 收益
        """)

    # 底部: EOD chain 详细面板 (保留旧版)
    st.divider()
    with st.expander("EOD 期权链详细 (Moomoo / Yfinance)"):
        _cfg_opt2 = load_config()
        _eod_opt2, _snap_opt2 = load_latest_eod_snapshot(_cfg_opt2)
        _sig_now2 = None
        if _r2 is not None and _r2["buy_signal"]:
            _sig_now2 = (_r2["buy_type"].replace(" ", "_")
                         if _r2["buy_type"] else "BUY_CALL")
        if _r2 is not None and _r2["exit_signal"]:
            _sig_now2 = _sig_now2 or "EXIT"
        if _is_straddle_now:
            _sig_now2 = "STRADDLE"
        _render_options_section(_eod_opt2, _snap_opt2, last_close, eff_bp090,
                                oi_adj_bp090=oi_adj_bp090,
                                gc_gld_ratio=gc_gld_ratio,
                                today_sgt=today_sgt, current_signal=_sig_now2,
                                straddle_active=_is_straddle_now,
                                straddle_reason=(_uni_today["chosen_reason"]
                                                  if _is_straddle_now else ""),
                                rv_val=rv)

    # ════════════════════════════════════════════════════════════════
    # v3.7.89 持仓管理重构: 4 节合并为 2 节 (按用户需求精简)
    # (1) 今日盘中触发  (2) 历史未平仓信号 (含真实期权 entry/current)
    # 删除: 盘中触发实盘模拟 / 日线信号模拟持仓 / 实盘持仓手动录入
    # ════════════════════════════════════════════════════════════════

    # ── (1) 今日盘中触发 ──
    st.divider()
    st.subheader(f"⚡ 今日盘中触发 ({today_sgt})")
    _today = pd.Timestamp(today_sgt).normalize()
    _today_log = _intra_log_asset[
        pd.to_datetime(_intra_log_asset["date"]).dt.normalize() == _today
    ] if len(_intra_log_asset) else _intra_log_asset
    if len(_today_log):
        _today_chosen = (sig_df.loc[_today, "chosen"]
                          if _today in sig_df.index else "")
        _is_strad_today = bool(sig_df.loc[_today, "straddle_signal"]
                                if _today in sig_df.index else False)
        _rows1 = []
        from core.paper_positions import price_strategy_at as _price_strat
        # buy_type 是日线策略, 不是 chosen
        _today_buy_type = (sig_df.loc[_today, "buy_type"]
                            if _today in sig_df.index else None) or ""
        # 当日 ETF daily OHLC (用于插值)
        _O_today = float(close_d.get(_today, 0))
        _C_today = float(close_d.get(_today, 0))
        if _today in close_d.index:
            try:
                # 真实 daily Open
                _gld_csv = pd.read_csv(
                    f"/Users/yhdong/Gold/data/raw/market/{asset_key.lower()}.csv",
                    index_col=0, parse_dates=True)
                if _today in _gld_csv.index:
                    _O_today = float(_gld_csv.loc[_today, "Open"])
                    _C_today = float(_gld_csv.loc[_today, "Close"])
                    _H_today = float(_gld_csv.loc[_today, "High"])
                    _L_today = float(_gld_csv.loc[_today, "Low"])
                else:
                    _H_today = float(high_d.get(_today, _C_today))
                    _L_today = float(low_d.get(_today, _C_today))
            except Exception:
                _H_today = _L_today = _C_today
        for _, r in _today_log.iterrows():
            _t = pd.Timestamp(r["trigger_time"])
            _h = _t.hour + _t.minute/60.0
            _is_rth = 9.5 <= _h <= 16.0
            _ul = float(r["price"])
            if _is_strad_today:
                _strat = "STRADDLE"
            elif _is_rth:
                _strat = _today_buy_type if _today_buy_type else "SPOT"
            else:
                _strat = "FUTURES_LONG"
            _opt_code = ""; _opt_p = "—"
            if _strat in ("BUY CALL", "SELL PUT", "STRADDLE"):
                _pricing = _price_strat(asset_key, _strat, _today, _t, _ul,
                                          _O_today, _C_today,
                                          _H_today, _L_today)
                if _pricing["legs"]:
                    _opt_code = _pricing["source"]
                    if _strat == "SELL PUT":
                        _opt_p = f"收${_pricing['entry_price']:.2f}"
                    else:
                        _opt_p = f"${_pricing['entry_price']:.2f}"
                else:
                    _opt_code = "(kline_db 无)"
            elif _strat == "FUTURES_LONG":
                _opt_code = f"{_futures_ticker} 多头"
                _opt_p = f"${_ul:.2f}"
            _rows1.append({
                "时间(ET)": _t.strftime("%m-%d %H:%M"),
                "信号": r["side"],
                "策略": _strat,
                "Underlying": f"${_ul:.2f}",
                "期权": _opt_code,
                "入场价": _opt_p,
                "TF": r.get("timeframe", "?"),
            })
        st.dataframe(pd.DataFrame(_rows1).iloc[::-1],
                      use_container_width=True, hide_index=True)
        st.caption("RTH (09:30-16:00 ET) 触发用日线 buy_type; "
                   "非 RTH 走期货多头 (期权未开盘). "
                   "期权价 = kline_db EOD OHLC + spot 比例插值 (远比 BS+假IV 准).")
    else:
        st.caption(f"今日 {today_sgt} 无盘中触发")

    # ── (2) 历史未平仓信号 + (3) 近一月已平期权模拟 共用单一构建 ──
    # 数据源: sig_df.buy_signal + detect_straddle/short_vol (event-mode 跟主图一致)
    # 平仓判定:
    #   STRADDLE → +14d 后 close 定时平 (long vol 衰减期)
    #   SHORT_VOL → +30d 后 close 定时平 (theta 收满)
    #   方向性 (BUY CALL/SELL PUT) → 找 intraday log 首个 EXIT trigger
    # 已平 → 期权模拟; 未到平仓日/无 EXIT → 历史未平仓
    from core.events import (detect_straddle_signal as _det_strad_uni,
                              detect_short_vol_signal as _det_sv_uni)
    from core.paper_positions import (
        price_strategy_at as _price_uni,
        _load_kline_db as _kdb_uni,
        simulate_option_exit as _sim_exit,
    )
    _kdb_u = _kdb_uni()
    _today_dt_u = pd.Timestamp(today_sgt).normalize()
    _cur_spot_u = float(close_d.iloc[-1])
    _gld_csv_u = pd.read_csv(
        f"/Users/yhdong/Gold/data/raw/market/{asset_key.lower()}.csv",
        index_col=0, parse_dates=True)
    _rv_s_u = features.loc[close_d.index, "rv_10d"] if "rv_10d" in features.columns else pd.Series(dtype=float)
    _bt_window_start = pd.Timestamp(today_sgt) - timedelta(days=60)
    _u_dates = sig_df.index[sig_df.index >= _bt_window_start]
    try:
        _strad_u = _det_strad_uni(_rv_s_u, _u_dates,
                                     rv_pctile=rv_pctile, asset=asset_key)
        _sv_u = _det_sv_uni(_rv_s_u, rv_pctile, _u_dates, regime=regime)
    except Exception:
        _strad_u = pd.DataFrame(); _sv_u = pd.DataFrame()
    _log_u = _intra_log_asset.copy() if len(_intra_log_asset) else _intra_log_asset
    if len(_log_u):
        _log_u["date"] = pd.to_datetime(_log_u["date"])
        _log_u["trigger_time"] = pd.to_datetime(_log_u["trigger_time"])
    _closed_recs = []; _open_recs = []
    for _du, _ru in sig_df.loc[_u_dates].iterrows():
        _is_strad_u = (len(_strad_u) > 0 and _du in _strad_u.index
                        and bool(_strad_u.loc[_du, "straddle_signal"]))
        _is_sv_u = (len(_sv_u) > 0 and _du in _sv_u.index
                     and bool(_sv_u.loc[_du, "short_vol_signal"]))
        if _is_strad_u: _strat = "STRADDLE"
        elif _is_sv_u: _strat = "SHORT_VOL"
        elif _ru.get("buy_signal", False):
            _strat = _ru.get("buy_type") or ""
        else: continue
        if not _strat: continue
        _entry_spot_u = float(close_d.get(_du, 0))
        if _entry_spot_u <= 0: continue
        if _du in _gld_csv_u.index:
            _eO = float(_gld_csv_u.loc[_du, "Open"])
            _eC = float(_gld_csv_u.loc[_du, "Close"])
            _eH = float(_gld_csv_u.loc[_du, "High"])
            _eL = float(_gld_csv_u.loc[_du, "Low"])
        else:
            _eO = _eC = _eH = _eL = _entry_spot_u
        _ent_pricing = _price_uni(asset_key, _strat, _du,
                                     _du + pd.Timedelta(hours=9, minutes=30),
                                     _eO, _eO, _eC, _eH, _eL,
                                     dte_target=(14 if _strat == "STRADDLE" else 30))
        if not _ent_pricing["legs"]:
            continue  # kline_db 无该日期权数据
        _ent_prem = _ent_pricing["entry_price"]
        _ent_legs = _ent_pricing.get("leg_prices", [])
        # v3.7.96: 真实期权退出规则 (50% profit / stop / expiry / 时间)
        _sim = _sim_exit(_ent_pricing, _du, _strat, _today_dt_u)
        _is_closed = _sim.get("is_closed", False)
        _gain_u = _sim.get("pnl_pct", 0.0)
        _exit_legs = _sim.get("leg_prices", [])
        # v3.7.97: 单腿价格显示 (用户偏好单腿 — 流动性高/价格更平滑)
        # SELL PUT: "Short P\$465: \$22.13 / Long P\$445: \$13.45 = 收 \$8.55"
        # STRADDLE: "Call \$X / Put \$Y = \$Z"
        # BUY CALL: "Call \$X"
        def _fmt_legs(legs_def, leg_prices):
            if not legs_def or not leg_prices:
                return "—"
            parts = []
            for (lab, code, K, qty), (lp_lab, p) in zip(legs_def, leg_prices):
                if "short" in lab:
                    parts.append(f"-P${K:.0f}@${p:.2f}")
                elif "put" in lab:
                    parts.append(f"+P${K:.0f}@${p:.2f}")
                elif "call" in lab:
                    parts.append(f"+C${K:.0f}@${p:.2f}")
            return " / ".join(parts)
        _ent_str = _fmt_legs(_ent_pricing["legs"], _ent_legs)
        if "SELL PUT" in _strat:
            _ent_str += f" → 收${_ent_prem:.2f}"
        elif "STRADDLE" in _strat or "SHORT_VOL" in _strat:
            _ent_str += f" → ${_ent_prem:.2f}"
        if _is_closed:
            _exit_d_u = _sim["exit_date"]
            _ex_val = _sim["exit_value"]
            _exit_label = f'{_exit_d_u.strftime("%m-%d")} ({_sim["exit_reason"]})'
            _exit_str = _fmt_legs(_ent_pricing["legs"], _exit_legs)
            if "SELL PUT" in _strat:
                _exit_str += f" → 平收${_ex_val:.2f}"
            else:
                _exit_str += f" → 平${_ex_val:.2f}"
            _exit_spot_u = float(close_d.get(_exit_d_u, _cur_spot_u))
        else:
            _ex_val = _sim.get("current_value", 0)
            _exit_label = f'OPEN ({_sim.get("hold_days", 0)}d)'
            _exit_str = _fmt_legs(_ent_pricing["legs"], _exit_legs)
            if "SELL PUT" in _strat:
                _exit_str += f" → 现${_ex_val:.2f}"
            else:
                _exit_str += f" → 现${_ex_val:.2f}"
            _exit_spot_u = _cur_spot_u
        _rec = {
            "信号日": _du.strftime("%m-%d"),
            "策略": _strat,
            "合约": _ent_pricing.get("source", "—"),
            "入场Spot": f"${_entry_spot_u:.2f}",
            "入场期权": _ent_str,
            "平/现Spot": f"${_exit_spot_u:.2f}",
            "平/现期权": _exit_str,
            "P&L%": f"{_gain_u:+.1f}%",
            "出场原因": _exit_label,
        }
        if _is_closed:
            _closed_recs.append(_rec)
        else:
            _open_recs.append(_rec)

    # 历史未平仓信号 (OPEN)
    st.divider()
    st.subheader(f"📊 历史未平仓信号 ({len(_open_recs)} 笔)")
    if _open_recs:
        st.dataframe(pd.DataFrame(_open_recs).iloc[::-1],
                      use_container_width=True, hide_index=True)
        st.caption("STRADDLE 还在 14d 内 / SHORT_VOL 还在 30d 内 / 方向性还没 EXIT trigger")
    else:
        st.caption("当前无未平仓信号")

    # 近一月期权模拟 (CLOSED)
    st.divider()
    st.subheader(f"🎯 近一月期权模拟 — 已平仓 ({len(_closed_recs)} 笔)")
    if _closed_recs:
        st.dataframe(pd.DataFrame(_closed_recs).iloc[::-1],
                      use_container_width=True, hide_index=True)
        _wins_c = sum(1 for r in _closed_recs
                       if float(r["P&L%"].rstrip("%")) > 0)
        _wr_c = _wins_c / len(_closed_recs) * 100
        _avg_c = sum(float(r["P&L%"].rstrip("%"))
                      for r in _closed_recs) / len(_closed_recs)
        c1, c2, c3 = st.columns(3)
        c1.metric("已平笔数", len(_closed_recs))
        c2.metric("胜率", f"{_wr_c:.0f}%")
        c3.metric("均 P&L", f"{_avg_c:+.2f}%")
    else:
        st.caption("近 60 天内无已平仓信号 (vol 都还在 hold 期内, 方向性无 EXIT trigger)")

    # ── (1) 日线简易回测 (180 天) ──
    st.divider()
    st.subheader("📈 日线简易回测 (180 天 · 伦敦金价位级)")
    _bt_180_start = pd.Timestamp(today_sgt) - timedelta(days=180)
    _bt_180_dates = sig_df.index[sig_df.index >= _bt_180_start]
    _bt_180_sig = sig_df.loc[_bt_180_dates]
    # v3.7.94: STRADDLE/SHORT_VOL 单独 detect (event-mode 跟主图一致)
    from core.events import (detect_straddle_signal as _det_strad_180,
                              detect_short_vol_signal as _det_sv_180)
    _rv_s_180 = features.loc[close_d.index, "rv_10d"] if "rv_10d" in features.columns else pd.Series(dtype=float)
    try:
        _strad_180 = _det_strad_180(_rv_s_180, _bt_180_dates,
                                       rv_pctile=rv_pctile, asset=asset_key)
        _sv_180 = _det_sv_180(_rv_s_180, rv_pctile, _bt_180_dates,
                                 regime=regime)
    except Exception:
        _strad_180 = pd.DataFrame(); _sv_180 = pd.DataFrame()
    _bt_recs = []
    _trades_idx = {t["entry_date"]: t for t in (trades or [])
                    if t.get("entry_date") is not None}
    for _d_b, _r_b in _bt_180_sig.iterrows():
        # v3.7.94: STRADDLE/SHORT_VOL 来自 _strad_180/_sv_180 (event-mode)
        _is_strad_b = (len(_strad_180) > 0 and _d_b in _strad_180.index
                        and bool(_strad_180.loc[_d_b, "straddle_signal"]))
        _is_sv_b = (len(_sv_180) > 0 and _d_b in _sv_180.index
                     and bool(_sv_180.loc[_d_b, "short_vol_signal"]))
        if _is_strad_b:
            _ch = "STRADDLE"
        elif _is_sv_b:
            _ch = "SHORT_VOL"
        elif _r_b.get("buy_signal", False):
            _ch = _r_b.get("buy_type", "") or ""
        else:
            continue
        if not _ch: continue
        _entry_spot = float(close_d.get(_d_b, 0))
        if _entry_spot <= 0: continue
        # 找匹配的 trade
        _t_match = _trades_idx.get(_d_b)
        if _t_match and _t_match.get("exit_date"):
            _exit_d = pd.Timestamp(_t_match["exit_date"])
            _exit_spot = float(_t_match.get("exit_price", 0))
            _exit_reason = _t_match.get("exit_type", "—")
        else:
            # 无 trade match (e.g. STRADDLE) → 入场后 5 天 close
            _later = close_d.index[close_d.index > _d_b]
            if len(_later) >= 5:
                _exit_d = _later[4]
                _exit_spot = float(close_d.get(_exit_d, _entry_spot))
                _exit_reason = "+5d"
            else:
                continue
        # P&L (spot 视角)
        _is_short = "SELL PUT" in _ch or "SHORT_VOL" in _ch
        _spot_chg = (_exit_spot / _entry_spot - 1) * 100
        _pnl = -_spot_chg if _is_short else _spot_chg
        if "STRADDLE" in _ch:
            _pnl = abs(_spot_chg) - 0.3  # long vol 估算
        _win = "✓" if _pnl > 0 else "✗"
        _bt_recs.append({
            "信号日": _d_b.strftime("%m-%d"),
            "策略": _ch.split("+")[0].strip(),
            "入场Spot": f"${_entry_spot:.2f}",
            "退出日": _exit_d.strftime("%m-%d"),
            "退出Spot": f"${_exit_spot:.2f}",
            "退出原因": _exit_reason,
            "Spot 涨跌": f"{_spot_chg:+.2f}%",
            "策略 P&L": f"{_pnl:+.2f}%",
            "结果": _win,
        })
    if _bt_recs:
        _bt_df = pd.DataFrame(_bt_recs).iloc[::-1]
        st.dataframe(_bt_df, use_container_width=True, hide_index=True)
        # 累计统计
        _wins = sum(1 for r in _bt_recs if r["结果"] == "✓")
        _wr = _wins / len(_bt_recs) * 100
        _avg = sum(float(r["策略 P&L"].rstrip("%")) for r in _bt_recs) / len(_bt_recs)
        _sum = sum(float(r["策略 P&L"].rstrip("%")) for r in _bt_recs)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("信号数", len(_bt_recs))
        c2.metric("胜率", f"{_wr:.0f}%")
        c3.metric("单笔均", f"{_avg:+.2f}%")
        c4.metric("累计", f"{_sum:+.1f}%")
    else:
        st.caption("180d 内无信号")

    # ── 模型信息 ──
    with st.expander("模型信息"):
        from core.signals_v2 import (BUY_BP, EXIT_BP, STOP_LOSS_PCT,
                                      PULLBACK_GAIN, PULLBACK_DD,
                                      CONSECUTIVE_STOP, MAX_HOLD_DAYS)
        gld_1h_info = (f"{gld_1h.index[0].strftime('%Y-%m-%d')} ~ "
                       f"{gld_1h.index[-1].strftime('%Y-%m-%d')} "
                       f"({len(gld_1h)} bars)"
                       if gld_1h is not None else "未加载")
        st.markdown(f"""
- **Band**: 日线模型 (20年训练, LSTM+Transformer Ensemble, Conformal 80%覆盖)
- **入场**: 日线 bp_low < {BUY_BP} (开窗) + 盘中真实触发 (Stoch RSI / MACD / KDJ 确认)
  - 入场价: 优先 log 当日代表价 (默认 worst); 没 log 退到收盘
- **退出 (优先级)**:
  1. **StopLoss**: 日内 low 跌破入场 -{STOP_LOSS_PCT}% → 即刻止损
  2. **BandExit**: bp_high > {EXIT_BP} → 优先 log EXIT 代表价, 兜底 bp090 阈值
  3. **Pullback**: 持仓期峰值涨幅 > {PULLBACK_GAIN}% 且回撤 >= {PULLBACK_DD}%
     (即持仓管理"止盈位"列那条线)
  4. **Timeout**: 持仓 ≥ {MAX_HOLD_DAYS} 天 (安全帽, 实际 2-5 天就走完)
- **风控**: 连续 {CONSECUTIVE_STOP} 笔止损熔断 (bp>0.50 恢复)
- **盘中数据**: {gld_1h_info}
""")


# ══════════════════════════════════════════════════════════
# 共享: 期权策略推荐
# ══════════════════════════════════════════════════════════
def _render_options_section(eod_df, snap_date, last_close, next_bp090,
                            oi_adj_bp090=0, gc_gld_ratio=None,
                            today_sgt=None, current_signal=None,
                            straddle_active=False, straddle_reason="",
                            rv_val=0):
    """渲染期权策略推荐 — 只推荐当日最优策略."""
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
    price_src = f"实时≈${current_gld:.1f}" if _rt else f"收盘 ${last_close:.2f}"

    st.caption(f"期权数据: EOD {snap_date} | {price_src} | 退出${eff_exit:.1f}")

    # ── 根据当前信号推荐最优策略 ──
    if current_signal == "EXIT":
        st.warning("**EXIT — 建议平仓**\n\n"
                   "- 持有 Call → 市价平仓\n"
                   "- 持有 Straddle → 如已盈利, 平仓锁利\n"
                   "- 暂停新开仓")
        return

    if straddle_active:
        # Straddle 推荐
        st.success(f"**推荐: Straddle (做多波动率)**\n\n"
                   f"触发: {straddle_reason}\n\n"
                   f"操作: 买入 ATM Call + ATM Put (同行权价, 同到期日)\n"
                   f"目标: 5天内波动 > 权利金成本\n"
                   f"退出: 事件结束后 (FOMC声明/OPEX结算) 或持有5天")

        # 显示 Straddle 成本
        result_call = get_strategy_table("BUY_CALL", current_gld, eff_exit,
                                          eod_df, use_live=True)
        result_put = get_strategy_table("SELL_PUT", current_gld, eff_exit,
                                         eod_df, use_live=True)
        if result_call.get("single_leg"):
            atm = [r for r in result_call["single_leg"] if "中性" in r.get("策略","")]
            if atm:
                st.markdown(f"ATM Call: {atm[0].get('合约','')} 成本{atm[0].get('成本','')}")
        return

    if current_signal in ("BUY_CALL", "SELL_PUT"):
        result = get_strategy_table(current_signal, current_gld, eff_exit,
                                     eod_df, use_live=True)

        if current_signal == "BUY_CALL":
            if result.get("rec"):
                st.success(result["rec"])
            else:
                st.success("**BUY CALL 信号 — 3种策略可选**")

            st.markdown(f"可选: 单腿Call / Bull Call Spread / "
                        f"{'高IV建议价差' if rv_val > 25 else '低IV建议单腿'}")

            # 只显示推荐的那个
            if rv_val > 28 and result.get("spread"):
                st.markdown("**推荐: Bull Call Spread** (IV偏高, 对冲theta)")
                st.dataframe(pd.DataFrame(result["spread"]),
                             use_container_width=True, hide_index=True)
            elif result.get("single_leg"):
                st.markdown("**推荐: Long Call** (IV适中)")
                st.dataframe(pd.DataFrame(result["single_leg"]),
                             use_container_width=True, hide_index=True)

        else:  # SELL_PUT
            result_put = get_strategy_table("SELL_PUT", current_gld, eff_exit,
                                             eod_df, use_live=True)
            st.success("**SELL PUT 信号 (高IV) — 推荐 Bull Put Spread**")
            if result_put.get("spread"):
                st.dataframe(pd.DataFrame(result_put["spread"]),
                             use_container_width=True, hide_index=True)
    else:
        # 观望区 — 简要说明
        st.info(f"当前观望区 (bp 0.30~0.90) — 等待信号触发\n\n"
                f"若触发买入: {'建议价差(IV偏高)' if rv_val > 25 else '可选单腿或价差'}\n"
                f"若波动率压缩+临近事件: 考虑 Straddle")


# ══════════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════════
def main():
    today_sgt = get_today_sgt()
    st.title(f"贵金属交易仪表板  ({today_sgt})")

    # 自动检测并更新市场数据
    cfg_refresh = load_config()
    with st.spinner("检测数据更新..."):
        refresh_results = auto_refresh_market_data(cfg_refresh)
        refreshed = [f"{t}: {s}" for t, s in refresh_results if "更新" in s]

        # 全量重建特征 + 扩展 OOS 预测
        try:
            n_feat, feat_msg = update_features_full(cfg_refresh)
            refresh_results.append(("特征", feat_msg))
            if n_feat > 0:
                refreshed.append(f"特征: {feat_msg}")
            n_new, oos_msg = extend_oos_predictions(cfg_refresh, asset="gld")
            refresh_results.append(("GLD OOS", oos_msg))
            if n_new > 0:
                refreshed.append(f"GLD OOS: {oos_msg}")
            # SLV 同步扩展
            try:
                n_slv, slv_msg = extend_oos_predictions(cfg_refresh, asset="slv")
                refresh_results.append(("SLV OOS", slv_msg))
                if n_slv > 0:
                    refreshed.append(f"SLV OOS: {slv_msg}")
            except Exception as e:
                refresh_results.append(("SLV OOS", f"失败: {e}"))
        except Exception as e:
            refresh_results.append(("OOS预测", f"失败: {e}"))

        if refreshed:
            load_all.clear()
            st.toast("数据已更新: " + " | ".join(refreshed), icon="✅")

    # ── 资产选择 ──
    st.sidebar.header("设置")
    asset = st.sidebar.selectbox("资产", ["GLD (黄金)", "SLV (白银)"], index=0)
    asset_key = "GLD" if "GLD" in asset else "SLV"

    # v3.7.63: 模块化 — GLD/SLV 共用 pipeline, 仅参数不同
    def _load_asset_pipeline(asset_key: str):
        """统一加载 GLD/SLV: (df, oos, regime, rv_pctile, futures_ratio, usdcny)
        regime 复用黄金 (宏观对两资产都适用).
        """
        # 共享: regime + usdcny + (GLD 自己) + ratios
        (gld_df, gld_oos, regime_, rv_pct_gld,
         gc_gld_r, usdcny_r, si_slv_r) = load_all()
        if asset_key == "GLD":
            return (gld_df, gld_oos, regime_, rv_pct_gld,
                    gc_gld_r, usdcny_r, si_slv_r)
        # SLV: 切数据源
        _slv = os.path.join(cfg_refresh["data_root"], "raw/market/slv.csv")
        _slv_oos = os.path.join(cfg_refresh["data_root"],
                                  "models/dl_range_slv_oos.parquet")
        _slv_feat = os.path.join(cfg_refresh["data_root"],
                                   "processed/features_slv.parquet")
        if not (os.path.exists(_slv) and os.path.exists(_slv_oos)):
            st.error("白银数据未找到 — 请先训练白银模型")
            st.stop()
        slv_df = pd.read_csv(_slv, index_col=0, parse_dates=True)
        slv_oos = pd.read_parquet(_slv_oos)
        if os.path.exists(_slv_feat):
            _f = pd.read_parquet(_slv_feat)
            rv_pct_slv = (_f["rv_10d"].rolling(252, min_periods=60)
                          .rank(pct=True) if "rv_10d" in _f.columns
                          else rv_pct_gld)
        else:
            rv_pct_slv = rv_pct_gld
        # SLV 用 si_slv_ratio (而非 gc_gld)
        return (slv_df, slv_oos, regime_, rv_pct_slv,
                si_slv_r, usdcny_r, si_slv_r)

    with st.spinner(f"加载{asset_key}数据..."):
        (gld, range_df, regime, rv_pctile, gc_gld_ratio, usdcny_rate,
         si_slv_ratio) = _load_asset_pipeline(asset_key)

    close, high, low = gld["Close"], gld["High"], gld["Low"]

    # 信号计算
    upper_band, lower_band, bp = build_band(
        range_df, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp_dates = bp.dropna().index
    bp_s = bp.reindex(bp_dates)
    rv_p = rv_pctile.reindex(bp_dates)
    is_bull = regime.reindex(bp_dates) == "Bull"
    buy_call, sell_put, exit_sig = generate_signals(bp_s, rv_p, is_bull,
                                                         asset=asset_key)

    # last_date = 最后一天有模型预测的日期 (用于 Band / 信号)
    # price_date = 最新的现货收盘价日期 (用于顶部指标 / 图表)
    # 二者可能不等 (例如 SLV 模型未重训, 但 slv.csv 已拉到最新)
    last_date = bp_dates[-1]
    price_date = close.index[-1]
    last_close = close.iloc[-1]  # 显示用: 最新实际收盘价
    last_bp = bp_s.get(last_date, 0)
    last_regime = regime.get(last_date, regime.iloc[-1] if len(regime) else "?")
    last_rv = rv_p.get(last_date, 0)
    _pred_stale_days = (price_date - last_date).days

    mode = st.sidebar.radio("模式", ["盘中信号", "今日预测", "历史回看", "回测分析"])

    # ── 阈值重测状态 (v3.7.31) ──
    with st.sidebar.expander("⚙️ 参数重测状态", expanded=False):
        from core.strategy_config import ASSET_CONFIGS, get_config
        from datetime import date as _date
        _today_d = pd.Timestamp.now().date()
        for _ast, _ac in ASSET_CONFIGS.items():
            if _ac.last_tuned:
                try:
                    _last = _date.fromisoformat(_ac.last_tuned)
                    _days = (_today_d - _last).days
                    _icon = "🟢" if _days < 30 else "🟡" if _days < 60 else "🔴"
                    st.caption(f"{_icon} **{_ast}**: {_ac.last_tuned} ({_days}d 前)")
                    if _days >= 30:
                        st.caption(f"   ⚠️ 建议月度重测")
                except Exception:
                    st.caption(f"⚪ {_ast}: 未校准")
        st.caption("命令行重测:")
        st.code("python scripts/monthly_retune.py", language="bash")

    # 预测过期警告 (价格比预测新 > 2 天)
    if _pred_stale_days > 2:
        st.sidebar.warning(
            f"⚠️ {asset_key} 预测停在 {last_date.date()}, "
            f"实际价格已到 {price_date.date()} ({_pred_stale_days}天差)\n\n"
            "请在侧边栏 '模型训练' 面板点击【重新训练模型】")

    # 数据状态 + 更新提醒
    with st.sidebar.expander("数据状态", expanded=False):
        st.caption(f"今日 (SGT): {today_sgt}")
        st.caption(f"{asset_key} 价格最新: {price_date.date()}")
        st.caption(f"{asset_key} 预测最新: {last_date.date()}")
        st.caption("💡 每次刷新页面自动更新行情 + 特征 + 预测")
        if st.button("🔄 立即刷新数据", key="btn_force_refresh",
                      help="清空缓存并重新拉取行情/特征/预测"):
            load_all.clear()
            st.rerun()
        for t, s in refresh_results:
            st.caption(f"{t}: {s}")

        # 手动数据更新提醒
        _cb_path = os.path.join(cfg_refresh["data_root"], "raw",
                                 "central_bank", "cb_features.csv")
        _etf_path = os.path.join(cfg_refresh["data_root"], "raw",
                                  "market", "gold_etf_holdings.csv")
        _stale = []
        for _fp, _name, _max_days in [(_cb_path, "央行购金", 45),
                                        (_etf_path, "黄金ETF", 35)]:
            if os.path.exists(_fp):
                _df = pd.read_csv(_fp, index_col=0, parse_dates=True)
                _age = (pd.Timestamp(str(today_sgt)) - pd.Timestamp(_df.index[-1])).days
                if _age > _max_days:
                    _stale.append(f"{_name} ({_age}天前)")
            else:
                _stale.append(f"{_name} (无数据)")
        if _stale:
            st.warning(f"需要更新: {', '.join(_stale)}\n\n"
                       "请下载 WGC 数据到 `Gold/data/download/` 后运行:\n"
                       "`python scripts/parse_wgc_data.py`")

    # ══ 模型训练状态 + 控制 (GLD + SLV) ══
    _data_root = cfg_refresh["data_root"]
    _assets = [("gld", "GLD 黄金"), ("slv", "SLV 白银")]

    # 侧边栏顶部提示: 任意 asset 训练中或过期都汇总显示
    _alerts = []
    for _ak, _alabel in _assets:
        if is_training(_ak):
            _alerts.append(f"🔄 {_alabel} 训练中 ({get_training_elapsed(_ak)})")
            continue
        _age = get_model_age_days(_data_root, _ak)
        if _age is None:
            _alerts.append(f"⚠️ {_alabel} 模型未训练")
        elif _age > DEFAULT_MAX_AGE_DAYS:
            _alerts.append(f"⚠️ {_alabel} 模型已 {_age:.0f} 天未训练")
    for _msg in _alerts:
        if "🔄" in _msg:
            st.sidebar.info(_msg)
        else:
            st.sidebar.warning(_msg)

    with st.sidebar.expander("模型训练", expanded=False):
        st.caption("训练内容: LSTM+Transformer Ensemble 22折 Walk-Forward (配置 A)")
        st.caption("预计耗时: 40~60 分钟 (MPS)")
        st.divider()

        for _ak, _alabel in _assets:
            st.markdown(f"**{_alabel}**")
            _age = get_model_age_days(_data_root, _ak)
            _is_running = is_training(_ak)

            if _age is None:
                st.caption("状态: 未训练")
            else:
                st.caption(f"最后训练: {_age:.1f} 天前")

            if _is_running:
                st.caption(f"进行中: {get_training_elapsed(_ak)}")
                if st.button("🛑 停止", key=f"btn_stop_{_ak}"):
                    ok, msg = stop_training(_ak)
                    st.toast(msg, icon="✅" if ok else "❌")
                    st.rerun()
                with st.expander("训练日志 (最近30行)", expanded=False):
                    log = get_training_log(30, _ak)
                    st.code(log or "(日志为空)", language="text")
            else:
                _btn_label = "🔄 重新训练" if _age is not None else "▶️ 开始训练"
                if st.button(_btn_label, key=f"btn_train_{_ak}",
                             type="primary"):
                    ok, msg = start_training(_ak)
                    st.toast(msg, icon="✅" if ok else "❌")
                    if ok:
                        time.sleep(1)
                        st.rerun()
            st.divider()

    if mode == "回测分析":
        # ── 回测模式 ──
        st.divider()
        st.subheader(f"{asset_key} 策略回测 "
                     f"(真实策略: 盘中触发入场 + StopLoss/BandExit/Pullback)")

        # 加载 1h 数据 (用于止盈)
        _1h_file = "gld_1h.csv" if asset_key == "GLD" else "slv_1h.csv"
        _1h_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Gold", "data", "raw", "market", _1h_file)
        _1h_path = os.path.normpath(_1h_path)
        _asset_1h = pd.read_csv(_1h_path, index_col=0, parse_dates=True) \
            if os.path.exists(_1h_path) else None

        # 加载盘中触发 log (回测用真实代表价, 没记录的天退到阈值兜底)
        from core.intraday_triggers import (
            load_log as _bt_load, worst_of_day as _bt_worst)
        _bt_log_path = os.path.join(cfg_refresh["data_root"],
                                     "intraday_signal_log.parquet")
        _bt_log = _bt_load(_bt_log_path)
        _bt_log_a = _bt_log[_bt_log["asset"] == asset_key] \
            if len(_bt_log) else _bt_log
        _bt_buy_lk = _bt_worst(_bt_log_a, "BUY") \
            if len(_bt_log_a) else pd.DataFrame()
        _bt_exit_lk = _bt_worst(_bt_log_a, "EXIT") \
            if len(_bt_log_a) else pd.DataFrame()

        bt_fig, bt_summary, bt_by_mode = generate_backtest_chart(
            close, high, low, bp_dates, upper_band, lower_band,
            buy_call, sell_put, exit_sig,
            regime, rv_pctile, gld_1h=_asset_1h,
            asset_key=asset_key,
            entry_log_lookup=_bt_buy_lk,
            exit_log_lookup=_bt_exit_lk)
        st.pyplot(bt_fig, use_container_width=True)

        # 按入场口径分别展示 (每口径一个独立表)
        if bt_by_mode:
            st.subheader("回测统计 (按入场口径拆分)")
            _mode_label_map = {
                "log_worst": "log 最差 (保守, 当日多次盘中触发取最不划算)",
                "log_best":  "log 最优 (可达, 当日盘中触发中最便宜的 1h close)",
                "log_first": "log 第一次 (开窗后首个盘中触发)",
                "close":     "收盘价 (信号日 close, 不挂限价)",
            }
            for _m_key in ["log_worst", "log_best", "log_first", "close"]:
                if _m_key not in bt_by_mode:
                    continue
                _m_rows = bt_by_mode[_m_key]
                st.markdown(f"**{_mode_label_map.get(_m_key, _m_key)}**")
                _df_m = pd.DataFrame([r for _, r in _m_rows])
                st.dataframe(_df_m, use_container_width=True,
                             hide_index=True)
            st.caption(
                "已移除 旧 `min(bp030, lo)` 口径 (假设挂单成交在日内 tick 最低, "
                "现实不可达). log_best 是真正可达的最优入场."
            )

        import io
        buf = io.BytesIO()
        bt_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
        buf.seek(0)
        st.download_button("下载回测图", buf.getvalue(),
                           file_name=f"backtest_{asset_key.lower()}.png",
                           mime="image/png")
        plt.close(bt_fig)

        st.caption(f"注: 回测基于标的({asset_key})价格变化, 非期权实际损益. "
                   "期权杠杆效应会放大实际收益/亏损.")
        return  # 回测模式不显示其他内容

    if mode == "盘中信号":
        _render_intraday_mode(close, high, low, upper_band, lower_band,
                              regime, rv_pctile, bp_dates, bp_s,
                              gc_gld_ratio, usdcny_rate, today_sgt,
                              asset_key=asset_key)
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
    # 今日预测模式: 标题用伦敦金/伦敦银 (现货/期货价), ETF 价格作为副信息
    _top_rt = None
    if mode == "今日预测":
        _top_ticker = "GC=F" if asset_key == "GLD" else "SI=F"
        _top_rt = _get_realtime_prices(_top_ticker)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        delta_pct = None
        if len(close) > 1:
            delta_pct = f"{(last_close / close.iloc[-2] - 1) * 100:+.2f}%"
        if _top_rt and _top_rt.get("gc_price", 0) > 0:
            _spot_label = "伦敦金" if asset_key == "GLD" else "伦敦银"
            _pfmt = ".1f" if asset_key == "GLD" else ".2f"
            st.metric(_spot_label, f"${_top_rt['gc_price']:{_pfmt}}",
                      delta=f"{asset_key} ${last_close:.2f} "
                            f"({delta_pct or '—'})")
        else:
            st.metric(asset_key, f"${last_close:.2f}", delta=delta_pct)
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

    # OI 因子仅对 GLD 有效 (期权快照数据来源于 GLD 期权链)
    if mode == "今日预测" and pred_u_pct is not None and asset_key == "GLD":
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

    # 价位换算: 主图改用伦敦金/伦敦银
    _gc_ticker_for_chart = "GC=F" if asset_key == "GLD" else "SI=F"
    _gc_rt_chart = _get_realtime_prices(_gc_ticker_for_chart)
    if _gc_rt_chart and _gc_rt_chart.get("gc_price", 0) > 0 \
            and last_close > 0:
        _spot_ratio_chart = _gc_rt_chart["gc_price"] / last_close
    elif gc_gld_ratio:
        _spot_ratio_chart = gc_gld_ratio
    else:
        _spot_ratio_chart = 1.0
    _spot_label_chart = "伦敦金" if asset_key == "GLD" else "伦敦银"

    # Straddle 触发日 + 信号去重 (与盘中信号页一致)
    _straddle_for_chart = None
    try:
        from core.events import detect_straddle_signal as _dst_chart
        from core.data import load_features as _lf_chart
        from core.signals_v2 import generate_daily_signals as _gds_chart
        from core.strategy_selector import (
            build_unified_signals as _bus_chart,
            dedupe_unified as _dedupe_chart,
        )
        from core.intraday_triggers import (
            load_log as _il_load_chart, worst_of_day as _il_worst_chart)

        _feat_chart = _lf_chart(load_config())
        _rv_chart = _feat_chart["rv_10d"] \
            if "rv_10d" in _feat_chart.columns else None
        if _rv_chart is not None:
            # v3.7.69: 今日预测 chart 跟盘中信号 chart 用相同参数
            # (asset 切点 + tech-score + close/high/low + short_vol_df)
            from core.events import detect_short_vol_signal as _dsv_chart
            _str_df = _dst_chart(_rv_chart, viz_dates,
                                  rv_pctile=rv_pctile,
                                  close=close, high=high, low=low,
                                  asset=asset_key)
            _short_vol_chart = _dsv_chart(_rv_chart, rv_pctile, viz_dates,
                                            regime=regime,
                                            close=close, high=high, low=low,
                                            asset=asset_key)
            _sig_chart = _gds_chart(close, high, low,
                                     upper_band, lower_band,
                                     regime, rv_pctile, asset=asset_key)
            _uni_raw_chart = _bus_chart(_sig_chart, _str_df,
                                         close, high, low,
                                         short_vol_df=_short_vol_chart)
            # 加载 log 以便取真实触发价做去重判断
            _log_chart = _il_load_chart(os.path.join(
                load_config()["data_root"],
                "intraday_signal_log.parquet"))
            _log_a_chart = _log_chart[_log_chart["asset"] == asset_key] \
                if len(_log_chart) else _log_chart
            _w_buy = _il_worst_chart(_log_a_chart, "BUY") \
                if len(_log_a_chart) else pd.DataFrame()
            _w_exit = _il_worst_chart(_log_a_chart, "EXIT") \
                if len(_log_a_chart) else pd.DataFrame()

            def _lp_chart(d, side):
                lk = _w_buy if side == "BUY" else _w_exit
                if len(lk) == 0:
                    return None
                d_n = pd.Timestamp(d).normalize()
                if d_n not in lk.index:
                    return None
                return float(lk.loc[d_n, "price"])

            _uni_dd_chart = _dedupe_chart(_uni_raw_chart, close,
                                           log_price_fn=_lp_chart,
                                           low_d=low)
            # 限制到 viz_dates
            _uni_dd_chart = _uni_dd_chart[
                _uni_dd_chart.index.isin(viz_dates)]

            # 用去重后的 chosen 重建 buy_call/sell_put/exit_sig
            _bc = pd.Series(False, index=close.index)
            _sp = pd.Series(False, index=close.index)
            _ex = pd.Series(False, index=close.index)
            _str_kept = []
            for d, r in _uni_dd_chart.iterrows():
                ch = r["chosen"]
                if ch == "BUY CALL":
                    _bc[d] = True
                elif ch == "SELL PUT":
                    _sp[d] = True
                elif "EXIT" in ch:
                    _ex[d] = True
                elif ch == "STRADDLE":
                    _str_kept.append(d)
            _straddle_for_chart = pd.DatetimeIndex(_str_kept) \
                if _str_kept else None
    except Exception:
        _straddle_for_chart = None

    # v3.7.60: 用 v2 + dedupe 生成 unified_markers (跟盘中信号 chart 一致)
    # 这样今日预测 chart 也能显示加仓 (4-29 谷底等)
    _unified_markers_chart = None
    try:
        _uni_dd_for_chart = locals().get("_uni_dd_chart")
        if _uni_dd_for_chart is not None and len(_uni_dd_for_chart) > 0:
            _viz_set = set(viz_dates)
            _unified_markers_chart = _uni_dd_for_chart[
                _uni_dd_for_chart.index.isin(_viz_set)]
    except Exception:
        pass

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
        oi_adj_bp030=oi_adj_bp030, oi_adj_bp090=oi_adj_bp090,
        asset_key=asset_key,
        spot_ratio=_spot_ratio_chart,
        spot_label=_spot_label_chart,
        straddle_dates=_straddle_for_chart,
        unified_markers=_unified_markers_chart)

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
        # v3.7.28: Stoch RSI + 波动率走势 合并为一个 sharex 图,
        # 与主图 viz_dates 同范围, 时间刻度对齐
        st.subheader(f"📈 {asset_key} Stoch RSI + 波动率走势 (与主图同范围)")
        from core.events import (detect_short_vol_signal as _dsv_pred_top,
                                  detect_straddle_signal as _dlv_pred_top)
        _feat_pred_top = load_features(load_config())
        _feat_pred_top = _feat_pred_top.reindex(close.index).ffill()
        _rv_chart_top = (_feat_pred_top["rv_10d"]
                          if "rv_10d" in _feat_pred_top.columns
                          else pd.Series(20, index=close.index))

        # Stoch RSI K/D
        _k_top, _d_top = compute_daily_stoch_rsi(close)
        _viz_idx = pd.DatetimeIndex(viz_dates)
        _k_win = _k_top.reindex(_viz_idx)
        _d_win = _d_top.reindex(_viz_idx)
        _rv_win = _rv_chart_top.reindex(_viz_idx)
        _rvp_win = rv_pctile.reindex(_viz_idx)

        # 信号窗口 (做多/做空波动率)
        _long_w = _dlv_pred_top(_rv_chart_top, _viz_idx, rv_pctile=rv_pctile, asset=asset_key)
        _short_w = _dsv_pred_top(_rv_chart_top, rv_pctile, _viz_idx, regime=regime)

        fig_combo, (ax_st, ax_rv1, ax_rv2) = plt.subplots(
            3, 1, figsize=(18, 7), sharex=True,
            gridspec_kw={"height_ratios": [1.5, 2, 1], "hspace": 0.1})

        # Stoch RSI
        ax_st.axhspan(80, 100, color="#E53935", alpha=0.10)
        ax_st.axhspan(0, 20, color="#43A047", alpha=0.10)
        ax_st.axhline(80, color="#E53935", lw=0.6, ls="--", alpha=0.5)
        ax_st.axhline(20, color="#43A047", lw=0.6, ls="--", alpha=0.5)
        ax_st.plot(_viz_idx, _k_win.values, color="#1E88E5", lw=1.2, label="K")
        ax_st.plot(_viz_idx, _d_win.values, color="#FB8C00", lw=1.0, label="D")
        ax_st.set_ylim(-2, 102)
        ax_st.set_ylabel("Stoch RSI")
        ax_st.legend(loc="upper left", fontsize=8)
        ax_st.grid(alpha=0.3)
        # 当前 K/D 状态文字
        _last_k_top = _k_win.dropna().iloc[-1] if _k_win.dropna().size else None
        _last_d_top = _d_win.dropna().iloc[-1] if _d_win.dropna().size else None
        if _last_k_top is not None:
            zone_top = ("超买" if _last_k_top >= 80 else
                        "超卖" if _last_k_top <= 20 else "中性")
            ax_st.text(0.99, 0.92,
                        f"当前 K={_last_k_top:.0f} D={_last_d_top:.0f} ({zone_top})",
                        transform=ax_st.transAxes, ha="right", va="top",
                        fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.85))

        # RV
        ax_rv1.plot(_viz_idx, _rv_win.values, color="#5B6BFF", lw=1.2, label="RV 10d")
        ax_rv1.axhline(20, color="#1976D2", lw=0.5, ls=":", alpha=0.6)
        ax_rv1.axhline(25, color="#FF6F00", lw=0.5, ls=":", alpha=0.6)
        ax_rv1.axhline(40, color="#B71C1C", lw=0.5, ls=":", alpha=0.5)
        # 信号窗口着色 (黄=做多, 橙=做空)
        for d, r in _long_w.iterrows():
            if r["straddle_signal"]:
                ax_rv1.axvspan(d, d + timedelta(days=1),
                               alpha=0.15, color="#FFD700", lw=0)
        for d, r in _short_w.iterrows():
            if r["short_vol_signal"]:
                ax_rv1.axvspan(d, d + timedelta(days=1),
                               alpha=0.15, color="#FF6F00", lw=0)
        ax_rv1.set_ylabel("RV (%)")
        ax_rv1.legend(loc="upper left", fontsize=8)
        ax_rv1.grid(alpha=0.3)

        # RV %tile
        ax_rv2.fill_between(_viz_idx, 0, _rvp_win.values * 100,
                              color="purple", alpha=0.3)
        ax_rv2.plot(_viz_idx, _rvp_win.values * 100, color="purple", lw=0.8)
        ax_rv2.axhline(70, color="#FF6F00", lw=0.5, ls="--", alpha=0.6)
        ax_rv2.axhline(30, color="#1976D2", lw=0.5, ls="--", alpha=0.6)
        ax_rv2.set_ylabel("RV %tile")
        ax_rv2.set_ylim(0, 100)
        ax_rv2.grid(alpha=0.3)
        ax_rv2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_rv2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        # 顶部 ax_st 也显示 x 标签
        ax_st.tick_params(axis="x", labelbottom=True)
        plt.setp(ax_st.get_xticklabels(), fontsize=7, rotation=0)
        plt.setp(ax_rv2.get_xticklabels(), fontsize=8, rotation=0)

        plt.tight_layout()
        st.pyplot(fig_combo, use_container_width=True)
        plt.close(fig_combo)

        st.caption(f"做多波动率窗口: {int(_long_w['straddle_signal'].sum())} 天 | "
                    f"做空波动率窗口: {int(_short_w['short_vol_signal'].sum())} 天 "
                    f"(在主图同范围内)")

        st.divider()
        c_a, c_b = st.columns(2)

        with c_a:
            st.subheader("5日区间预测")
            if pred_u_pct is not None:
                tu = last_close * (1 + pred_u_pct / 100)
                tl = last_close * (1 + pred_l_pct / 100)
                # 期货/现货等价 (实时比价优先, 兜底用近60日均值)
                _spot_ratio = None
                if _top_rt and _top_rt.get("gc_price", 0) > 0:
                    _spot_ratio = _top_rt["gc_price"] / last_close
                elif gc_gld_ratio:
                    _spot_ratio = gc_gld_ratio
                _spot_label_t = "伦敦金" if asset_key == "GLD" else "伦敦银"
                _pfmt_t = ".1f" if asset_key == "GLD" else ".2f"
                _tu_spot = f"${tu * _spot_ratio:{_pfmt_t}}" if _spot_ratio else "—"
                _tl_spot = f"${tl * _spot_ratio:{_pfmt_t}}" if _spot_ratio else "—"
                if oi_adj_upper is not None:
                    adj_u_pct = (oi_adj_upper / last_close - 1) * 100
                    adj_l_pct = (oi_adj_lower / last_close - 1) * 100
                    _adj_u_spot = (f"${oi_adj_upper * _spot_ratio:{_pfmt_t}}"
                                   if _spot_ratio else "—")
                    _adj_l_spot = (f"${oi_adj_lower * _spot_ratio:{_pfmt_t}}"
                                   if _spot_ratio else "—")
                    st.markdown(f"""
| 指标 | 模型预测 ({asset_key}) | {_spot_label_t} | OI修正后 ({asset_key}) | OI修正 ({_spot_label_t}) |
|------|---------|---------|---------|---------|
| 预测日期 | {last_date.date()} (基于{today_sgt}) | | | |
| 上界 | ${tu:.2f} (+{pred_u_pct:.1f}%) | {_tu_spot} | **${oi_adj_upper:.2f}** ({adj_u_pct:+.1f}%) | **{_adj_u_spot}** |
| 下界 | ${tl:.2f} ({pred_l_pct:.1f}%) | {_tl_spot} | **${oi_adj_lower:.2f}** ({adj_l_pct:+.1f}%) | **{_adj_l_spot}** |
""")
                else:
                    st.markdown(f"""
| 指标 | {asset_key} (USD) | {_spot_label_t} (USD/oz) |
|------|------|------|
| 预测日期 | {last_date.date()} (基于{today_sgt}) | |
| 上界 | **${tu:.2f}** (+{pred_u_pct:.1f}%) | **{_tu_spot}** |
| 下界 | **${tl:.2f}** ({pred_l_pct:.1f}%) | **{_tl_spot}** |
| 区间宽度 | ${tu - tl:.2f} ({pred_u_pct - pred_l_pct:.1f}%) | |
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

            # 实时行情 (5分钟缓存): GLD→GC=F, SLV→SI=F
            _conv_ticker = "GC=F" if asset_key == "GLD" else "SI=F"
            rt = _get_realtime_prices(_conv_ticker)
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

            _is_gold_tbl = asset_key == "GLD"
            _etf_col = "GLD (USD)" if _is_gold_tbl else "SLV (USD)"
            _spot_col = "伦敦金现 XAU (USD/oz)" if _is_gold_tbl else "伦敦银现 XAG (USD/oz)"
            _fut_col = "纽约金 COMEX (USD/oz)" if _is_gold_tbl else "纽约银 COMEX (USD/oz)"
            _shfe_col = "沪金 AU (CNY/g)" if _is_gold_tbl else "沪银 AG (CNY/kg)"
            st.markdown(f"""
| 价位 | {_etf_col} | {_spot_col} | {_fut_col} | {_shfe_col} |
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
                   f"期货/ETF={_ratio:.4f} (近60日均值)")
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
                    f"- {asset_key} 期权有**月度**到期 (每月第三个周五) "
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

        # v3.7.28: 波动率走势已合并到主图下方 sharex 图 (Stoch RSI + RV + RV%tile)

        # ── 前瞻分析: 关键日程 + 信号判断 ──
        st.divider()
        st.subheader("前瞻分析")

        from core.events import (get_all_events, detect_straddle_signal,
                                  detect_short_vol_signal,
                                  days_to_next_event)
        _feat_pred = load_features(load_config())
        _feat_pred = _feat_pred.reindex(close.index).ffill()
        _rv_pred = _feat_pred["rv_10d"] if "rv_10d" in _feat_pred.columns \
            else pd.Series(20, index=close.index)

        # 近期事件 (含期货交割日)
        _asset_ev = "gold" if asset_key == "GLD" else "silver"
        _d_fomc, _, _fomc_d = days_to_next_event(last_date, "FOMC", _asset_ev)
        _d_opex, _, _opex_d = days_to_next_event(last_date, "OPEX", _asset_ev)
        _d_nfp, _, _nfp_d = days_to_next_event(last_date, "NFP", _asset_ev)
        _d_fut, _, _fut_d = days_to_next_event(last_date, "FUT_EXP", _asset_ev)

        # 做多波动率 + 做空波动率 检测 (v3.7.47 传 close/high/low 启用技术 score)
        _st_today = detect_straddle_signal(
            _rv_pred, pd.DatetimeIndex([last_date]),
            rv_pctile=rv_pctile,
            close=close, high=high, low=low, asset=asset_key)
        _is_straddle_pred = _st_today["straddle_signal"].iloc[0] \
            if len(_st_today) > 0 else False
        _straddle_reason_pred = _st_today["straddle_reason"].iloc[0] \
            if _is_straddle_pred else ""

        # v3.7.49: 传 close/high/low 启用技术 score 模式
        _sv_today = detect_short_vol_signal(
            _rv_pred, rv_pctile, pd.DatetimeIndex([last_date]),
            regime=regime,
            close=close, high=high, low=low, asset=asset_key)
        _is_short_vol_pred = _sv_today["short_vol_signal"].iloc[0] \
            if len(_sv_today) > 0 else False
        _short_vol_reason_pred = _sv_today["short_vol_reason"].iloc[0] \
            if _is_short_vol_pred else ""

        # 宏观指标
        _dxy = _feat_pred["dxy_ret_5d"].get(last_date, 0) \
            if "dxy_ret_5d" in _feat_pred.columns else 0
        _vix = _feat_pred["vix_level"].get(last_date, 0) \
            if "vix_level" in _feat_pred.columns else 0
        _ry = _feat_pred["real_yield_10y"].get(last_date, 0) \
            if "real_yield_10y" in _feat_pred.columns else 0
        _rv_val = _rv_pred.get(last_date, 20)

        col_outlook, col_macro = st.columns(2)
        with col_outlook:
            st.markdown("**未来关键日程**")
            events_5d = get_all_events(
                (last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                (last_date + timedelta(days=10)).strftime("%Y-%m-%d"),
                asset=_asset_ev)
            if events_5d:
                for ev_d, ev_t, ev_l in events_5d:
                    days_away = (ev_d - last_date).days
                    st.markdown(f"- **{ev_d.strftime('%m/%d')} {ev_l}** ({days_away}天后)")
            else:
                st.markdown("- 未来10天无重大事件")

            if _is_straddle_pred and _is_short_vol_pred:
                _l_score = int(_st_today["straddle_score"].iloc[0])
                _s_score = int(_sv_today["short_vol_score"].iloc[0])
                st.warning(f"**波动率信号冲突** L={_l_score} / S={_s_score}\n\n"
                           f"做多: {_straddle_reason_pred}\n\n"
                           f"做空: {_short_vol_reason_pred}\n\n"
                           "建议按 score 较高方向操作")
            elif _is_straddle_pred:
                # IV/RV 比率作为风险参考 (不调 P&L)
                # 实证 GLD GVZ FOMC 后 mean -0.1%, IV crush 不显著
                from core.iv_crush import (crush_risk_label,
                                            RATIO_HIGH_THRESHOLD,
                                            RATIO_MEDIUM_THRESHOLD)
                _gvz_now = (_feat_pred["gvz"].get(last_date, np.nan)
                            if "gvz" in _feat_pred.columns else np.nan)
                _iv_info = ""
                if not np.isnan(_gvz_now) and _rv_val > 0:
                    _ratio = _gvz_now / _rv_val
                    _level, _desc = crush_risk_label(_ratio)
                    _iv_info = (
                        f"\n\n📊 **IV / RV 比率**: GVZ={_gvz_now:.1f}, "
                        f"RV={_rv_val:.1f}, 比率={_ratio:.2f} ({_level}风险)\n"
                        f"  · 比率 < {RATIO_MEDIUM_THRESHOLD}: 无显著事件溢价\n"
                        f"  · {RATIO_MEDIUM_THRESHOLD} ~ {RATIO_HIGH_THRESHOLD}: 中等溢价\n"
                        f"  · > {RATIO_HIGH_THRESHOLD}: 显著事件溢价, crush 风险高\n"
                        f"  · GLD 实证 FOMC 后 GVZ 平均 -0.1% (远小于 SPX 30-60%)\n"
                        f"  · 当前比率仅供参考, 不调整 P&L"
                    )
                if _d_fomc <= 5:
                    _iv_info += (f"\n\n⚠️ 距 FOMC {_d_fomc} 天: 历史上 60% 概率"
                                 f" GLD IV 反而上涨, 40% 概率 crush > 5%. "
                                 "如担心尾部风险可提前 1 天平仓 / 改 Calendar Spread.")

                st.warning(f"**做多波动率信号**: {_straddle_reason_pred}\n\n"
                           "建议: 考虑做多波动率 (ATM Call+Put / 长 Strangle)"
                           + _iv_info)
            elif _is_short_vol_pred:
                st.warning(f"**做空波动率信号**: {_short_vol_reason_pred}\n\n"
                           "建议: 考虑做空波动率 (Iron Condor / 短 Strangle), "
                           "硬止损 IV 跳涨 30%")
            elif min(_d_fomc, _d_opex, _d_nfp, _d_fut) <= 5:
                st.info(f"临近事件日 — 关注波动率变化")

        with col_macro:
            st.markdown("**宏观环境**")
            dxy_comment = "美元走强→金价承压" if _dxy > 0.005 else \
                "美元走弱→金价利好" if _dxy < -0.005 else "中性"
            rv_comment = "低位(做多波动率机会)" if _rv_val < 20 else \
                "正常" if _rv_val < 35 else "高位(期权成本高)"
            st.markdown(f"""
- RV(10d): **{_rv_val:.1f}%** ({rv_comment})
- VIX: **{_vix:.1f}**
- DXY 5d: **{_dxy*100:+.2f}%** ({dxy_comment})
- 实际利率: **{_ry:.2f}%**
""")

        # 期权策略
        st.divider()
        st.subheader("交易工具推荐")

        # 按 v3.6.6 实证: BUY CALL 信号下期货 96% > 期权 73%
        if sig_type_viz == "BUY CALL":
            st.success(
                "📈 **首选: 期货多头 + 3% 止损** (实证胜率 96%, Sharpe 1.16)\n\n"
                "- 低 RV 期权虽便宜但仍要付 IV premium ~2-2.5%\n"
                "- 实际信号 Avg max_up +2.35%, 期权刚够 breakeven\n"
                "- 期货线性 P&L 无 theta/vega 损耗\n\n"
                "**备选**: Long Call (低 RV+临 FOMC 时考虑)")
        elif sig_type_viz == "SELL PUT":
            st.success(
                "💵 **首选: 期权 Sell Put** (实证胜率 100%)\n\n"
                "- 高 RV 反弹但持续性差, 期货仅 68% 胜率\n"
                "- 卖 Put 收 IV premium, 横盘+上涨都赢\n"
                "- 期货在震荡 regime 易被震出\n\n"
                "**禁用**: 期货多头 (本 RV regime 下 Sharpe 反向)")
        elif _is_straddle_pred:
            st.info("🌀 **首选: Long Straddle** (临事件做多波动率)")
        elif _is_short_vol_pred:
            st.warning("🔒 **首选: Iron Condor 16Δ/5Δ** (做空波动率)")

        st.subheader("期权策略详情")
        if eod_df is None:
            cfg = load_config()
            eod_df, snap_date = load_latest_eod_snapshot(cfg)

        # 传递 Straddle 状态
        _sig_for_opt = sig_type_viz
        if _is_straddle_pred:
            _sig_for_opt = "STRADDLE"
        _render_options_section(eod_df, snap_date, last_close,
                                next_bp090, oi_adj_bp090,
                                gc_gld_ratio, today_sgt, _sig_for_opt,
                                straddle_active=_is_straddle_pred,
                                straddle_reason=_straddle_reason_pred,
                                rv_val=_rv_val)

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
            asset_key: f"${close.get(d, 0):.2f}",
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
