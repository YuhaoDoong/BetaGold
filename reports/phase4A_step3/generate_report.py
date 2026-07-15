"""
Phase 4A 综合可视化报告 (V2 + DL 全方法对比)

图1: GLD 近一年日线 + Regime 着色
图2: V2 公允价格 (EMA20+macro) + 区间上下限 + 真实价格
图3: V2 交易信号 — 价格+区间+买卖点+收益对比
图4: 纯 Regime 策略收益对比
图5: 年度收益柱状图 (2016-2026)
图6: DL LSTM vs Transformer vs EMA 公允价格对比
图7: DL 区间预测 — 预测上下限 vs 实际波动
图8: 全方法对比汇总

用法:
    conda activate gold
    python reports/phase4A_step3/generate_report.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.weekly_range_signal_v2 import WeeklyRangeSignalV2

OUT_DIR = os.path.join(ROOT, "reports", "phase4A_step3")
MODEL_DIR = os.path.join(ROOT, "data", "models")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (18, 9),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.facecolor": "#fafafa",
})

REGIME_COLORS = {"Bull": "#4caf50", "Mixed": "#ff9800", "Bear": "#f44336"}


# ======================================================================
# Data loading
# ======================================================================

def load_all_data():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    gld_close = gld["Close"].rename("gld_close")

    common = features.index.intersection(gld_close.index)
    features, gld_close = features.loc[common], gld_close.loc[common]

    valid = features.notna().all(axis=1) & gld_close.notna()
    features, gld_close = features[valid], gld_close[valid]

    return features, gld_close


def compute_regime(features):
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    return RegimeClassifier().classify(features[feat_cols])["regime"]


def build_v2_signal(features, gld_close, regime):
    tw_usd = features["tw_usd"] if "tw_usd" in features.columns else None
    ry = features["real_yield_10y"] \
        if "real_yield_10y" in features.columns else None
    gvz = features["gvz"] if "gvz" in features.columns else None
    ret_20d = features["ret_20d"] if "ret_20d" in features.columns else None

    sig_gen = WeeklyRangeSignalV2(
        buy_zone=0.30, sell_zone=0.75,
        bull_min=0.3, bull_default=0.5,
        mixed_buy=0.5, mixed_default=0.3,
    )
    return sig_gen.generate(regime, gld_close, tw_usd=tw_usd,
                            real_yield_10y=ry, gvz=gvz, ret_20d=ret_20d)


def load_dl_data():
    """Load DL model results from parquet files."""
    dl_data = {}

    # DL LSTM fair value
    path = os.path.join(MODEL_DIR, "dl_fair_value_oos.parquet")
    if os.path.exists(path):
        dl_data["lstm_fv"] = pd.read_parquet(path)

    # DL Transformer fair value
    path = os.path.join(MODEL_DIR, "dl_transformer_fair_value_oos.parquet")
    if os.path.exists(path):
        dl_data["transformer_fv"] = pd.read_parquet(path)

    # DL Range V2
    path = os.path.join(MODEL_DIR, "dl_range_v2_oos.parquet")
    if os.path.exists(path):
        dl_data["range_v2"] = pd.read_parquet(path)

    return dl_data


# ======================================================================
# Helpers
# ======================================================================

def shade_regime(ax, regime):
    for r, color in REGIME_COLORS.items():
        mask = regime == r
        in_region = False
        start = None
        for i in range(len(mask)):
            if mask.iloc[i] and not in_region:
                start = mask.index[i]
                in_region = True
            elif not mask.iloc[i] and in_region:
                ax.axvspan(start, mask.index[i], alpha=0.12, color=color, lw=0)
                in_region = False
        if in_region:
            ax.axvspan(start, mask.index[-1], alpha=0.12, color=color, lw=0)


def find_trade_points(pos, threshold=0.1):
    buys, sells = [], []
    for i in range(1, len(pos)):
        prev, curr = pos.iloc[i-1], pos.iloc[i]
        if prev < threshold and curr >= threshold:
            buys.append(pos.index[i])
        elif prev >= threshold and curr < threshold:
            sells.append(pos.index[i])
    return buys, sells


def recent(df_or_series, start):
    return df_or_series[df_or_series.index >= start]


def regime_legend():
    return [
        Patch(facecolor=REGIME_COLORS["Bull"], alpha=0.25, label="Bull"),
        Patch(facecolor=REGIME_COLORS["Mixed"], alpha=0.25, label="Mixed"),
        Patch(facecolor=REGIME_COLORS["Bear"], alpha=0.25, label="Bear"),
    ]


def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ======================================================================
# Figure 1: GLD + Regime
# ======================================================================

def fig1_regime(gld_close, regime, recent_start):
    r = recent(regime, recent_start)
    p = recent(gld_close, recent_start).reindex(r.index)

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(p.index, p.values, color="#333", linewidth=1.8, label="GLD Close")
    shade_regime(ax, r)

    handles = [plt.Line2D([0], [0], color="#333", lw=1.8, label="GLD Close")]
    handles += regime_legend()
    ax.legend(handles=handles, loc="upper left", fontsize=12)

    ax.set_title("GLD Daily Price + Macro Regime (Recent 1 Year)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("GLD Price ($)", fontsize=13)
    format_date_axis(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_regime.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [1/8] Regime")


# ======================================================================
# Figure 2: Fair Value (EMA20 + macro shift) + Band
# ======================================================================

def fig2_fair_value_band(gld_close, signal, regime, recent_start):
    sig = recent(signal, recent_start)
    p = gld_close.reindex(sig.index)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 1],
                             sharex=True)

    ax1 = axes[0]
    ax1.plot(p.index, p.values, color="#333", linewidth=2.0,
             label="GLD Close (actual)", zorder=5)
    ax1.plot(sig.index, sig["fair_value"].values, color="#1565c0",
             linewidth=1.8, linestyle="--",
             label="Fair Value (EMA20 + macro)", zorder=4)
    ax1.plot(sig.index, sig["upper"].values, color="#d32f2f",
             linewidth=1.2, linestyle=":", label="Upper Bound", zorder=3)
    ax1.plot(sig.index, sig["lower"].values, color="#2e7d32",
             linewidth=1.2, linestyle=":", label="Lower Bound", zorder=3)
    ax1.fill_between(sig.index, sig["lower"].values, sig["upper"].values,
                     alpha=0.08, color="#1565c0")

    reg_sub = regime.reindex(sig.index).ffill()
    shade_regime(ax1, reg_sub)

    handles = [
        plt.Line2D([0], [0], color="#333", lw=2, label="GLD Close"),
        plt.Line2D([0], [0], color="#1565c0", lw=1.8, ls="--",
                   label="Fair Value"),
        plt.Line2D([0], [0], color="#d32f2f", lw=1.2, ls=":", label="Upper"),
        plt.Line2D([0], [0], color="#2e7d32", lw=1.2, ls=":", label="Lower"),
    ]
    handles += regime_legend()
    ax1.legend(handles=handles, loc="upper left", fontsize=11, ncol=2)
    ax1.set_title("V2 Fair Value (EMA20 + Macro Shift) + Dynamic Band",
                  fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=13)

    dev = (p - sig["fair_value"]) / sig["fair_value"]
    hw_mean = sig["half_width"].mean()
    ax1.annotate(
        f"FV deviation: mean={dev.mean():+.1%}, std={dev.std():.1%}\n"
        f"Avg half-width: {hw_mean:.1%}",
        xy=(0.72, 0.05), xycoords="axes fraction", fontsize=11,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax2 = axes[1]
    dev_pct = dev * 100
    colors = ["#2e7d32" if d < 0 else "#d32f2f" for d in dev_pct.values]
    ax2.bar(dev_pct.index, dev_pct.values, color=colors, alpha=0.6, width=1.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(-3, color="#2e7d32", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.axhline(3, color="#d32f2f", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_ylabel("Deviation (%)", fontsize=12)
    ax2.set_ylim(-8, 8)
    format_date_axis(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_fair_value_band.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [2/8] Fair Value Band")


# ======================================================================
# Figure 3: V2 Trading
# ======================================================================

def fig3_v2_trading(gld_close, regime, signal, recent_start):
    sig = recent(signal, recent_start)
    p = gld_close.reindex(sig.index)
    daily_ret = gld_close.pct_change().fillna(0).reindex(sig.index)

    pos = sig["position"]
    strat_ret = pos.shift(1).fillna(0) * daily_ret
    cum_strat = (1 + strat_ret).cumprod()
    cum_bh = (1 + daily_ret).cumprod()

    buys, sells = find_trade_points(pos)

    fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                             height_ratios=[3, 1, 2], sharex=True)

    ax1 = axes[0]
    ax1.plot(p.index, p.values, color="#333", linewidth=2.0,
             label="GLD Close", zorder=5)
    ax1.plot(sig.index, sig["fair_value"].values, color="#1565c0",
             linewidth=1.3, linestyle="--", label="Fair Value", zorder=4)
    ax1.plot(sig.index, sig["upper"].values, color="#d32f2f",
             linewidth=1, linestyle=":", alpha=0.7, zorder=3)
    ax1.plot(sig.index, sig["lower"].values, color="#2e7d32",
             linewidth=1, linestyle=":", alpha=0.7, zorder=3)
    ax1.fill_between(sig.index, sig["lower"].values, sig["upper"].values,
                     alpha=0.06, color="#1565c0")

    reg_sub = regime.reindex(sig.index).ffill()
    shade_regime(ax1, reg_sub)

    for b in buys:
        if b in p.index:
            ax1.scatter(b, p.loc[b], marker="^", color="#2e7d32", s=180,
                        zorder=6, edgecolors="black", linewidth=0.5)
    for s in sells:
        if s in p.index:
            ax1.scatter(s, p.loc[s], marker="v", color="#d32f2f", s=180,
                        zorder=6, edgecolors="black", linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], color="#333", lw=2, label="GLD Close"),
        plt.Line2D([0], [0], color="#1565c0", lw=1.3, ls="--",
                   label="Fair Value"),
        plt.Line2D([0], [0], color="#d32f2f", lw=1, ls=":", label="Upper"),
        plt.Line2D([0], [0], color="#2e7d32", lw=1, ls=":", label="Lower"),
        plt.Line2D([0], [0], marker="^", color="#2e7d32", ls="", ms=12,
                   label="Buy (enter)"),
        plt.Line2D([0], [0], marker="v", color="#d32f2f", ls="", ms=12,
                   label="Sell (exit)"),
    ]
    handles += regime_legend()
    ax1.legend(handles=handles, loc="upper left", fontsize=10, ncol=3)
    ax1.set_title("V2: Adaptive Range Trading (EMA20 + Macro + Regime)",
                  fontsize=16, fontweight="bold")
    ax1.set_ylabel("GLD Price ($)", fontsize=13)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    pos_colors = ["#2e7d32" if p > 0.5 else "#ff9800" if p > 0
                  else "#bdbdbd" for p in pos.values]
    ax2.bar(pos.index, pos.values, color=pos_colors, alpha=0.5, width=1.5)
    ax2.set_ylabel("Position", fontsize=11, color="#333")
    ax2.set_ylim(-0.1, 1.3)

    bp = sig["band_position"].clip(-0.5, 1.5)
    ax2_twin.plot(bp.index, bp.values, color="#9c27b0", linewidth=1, alpha=0.7)
    ax2_twin.axhline(0.30, color="#2e7d32", lw=0.8, ls=":", alpha=0.5)
    ax2_twin.axhline(0.75, color="#d32f2f", lw=0.8, ls=":", alpha=0.5)
    ax2_twin.set_ylabel("Band Position", fontsize=11, color="#9c27b0")
    ax2_twin.set_ylim(-0.5, 1.5)

    ax3 = axes[2]
    ax3.plot(cum_bh.index, cum_bh.values, color="#9e9e9e", linewidth=2.5,
             linestyle="--", label="Buy & Hold")
    ax3.plot(cum_strat.index, cum_strat.values, color="#1565c0", linewidth=2.5,
             label="V2: Adaptive Range")

    reg_pos = (reg_sub == "Bull").astype(float)
    reg_ret = reg_pos.shift(1).fillna(0) * daily_ret
    cum_reg = (1 + reg_ret).cumprod()
    ax3.plot(cum_reg.index, cum_reg.values, color="#ff9800", linewidth=2,
             linestyle="-.", label="Pure Regime (Bull=1)")

    ax3.axhline(1.0, color="black", linewidth=0.5)

    for cum, lbl, color, offset in [
        (cum_bh, "B&H", "#9e9e9e", 0.03),
        (cum_strat, "V2", "#1565c0", 0.0),
        (cum_reg, "Regime", "#ff9800", -0.03),
    ]:
        final = cum.iloc[-1]
        ax3.annotate(f"{lbl}: {final-1:+.1%}", xy=(cum.index[-1], final),
                     fontsize=11, fontweight="bold", color=color,
                     xytext=(10, 0), textcoords="offset points")

    ax3.legend(loc="upper left", fontsize=12)
    ax3.set_ylabel("Cumulative Return", fontsize=12)
    format_date_axis(ax3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_v2_trading.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [3/8] V2 Trading")


# ======================================================================
# Figure 4: Pure Regime Strategy
# ======================================================================

def fig4_pure_regime(gld_close, regime, recent_start):
    r = recent(regime, recent_start)
    p = recent(gld_close, recent_start).reindex(r.index)
    daily_ret = gld_close.pct_change().fillna(0).reindex(r.index)

    pos_bull = (r == "Bull").astype(float)
    pos_bm = pd.Series(0.0, index=r.index)
    pos_bm[r == "Bull"] = 1.0
    pos_bm[r == "Mixed"] = 0.5

    cum_bh = (1 + daily_ret).cumprod()
    cum_bull = (1 + pos_bull.shift(1).fillna(0) * daily_ret).cumprod()
    cum_bm = (1 + pos_bm.shift(1).fillna(0) * daily_ret).cumprod()

    buys, sells = find_trade_points(pos_bull)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[2, 1],
                             sharex=True)

    ax1 = axes[0]
    ax1.plot(p.index, p.values, color="#333", linewidth=1.8,
             label="GLD Close")
    shade_regime(ax1, r)

    for b in buys:
        if b in p.index:
            ax1.scatter(b, p.loc[b], marker="^", color="#2e7d32", s=150,
                        zorder=5, edgecolors="black", linewidth=0.5)
    for s in sells:
        if s in p.index:
            ax1.scatter(s, p.loc[s], marker="v", color="#d32f2f", s=150,
                        zorder=5, edgecolors="black", linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], color="#333", lw=1.8, label="GLD Close"),
        plt.Line2D([0], [0], marker="^", color="#2e7d32", ls="", ms=10,
                   label="Bull start"),
        plt.Line2D([0], [0], marker="v", color="#d32f2f", ls="", ms=10,
                   label="Bull end"),
    ]
    handles += regime_legend()
    ax1.legend(handles=handles, loc="upper left", fontsize=11)
    ax1.set_title("Pure Regime Strategy (Recent 1 Year)",
                  fontsize=15, fontweight="bold")
    ax1.set_ylabel("GLD Price ($)", fontsize=12)

    ax2 = axes[1]
    ax2.plot(cum_bh.index, cum_bh.values, color="#9e9e9e", linewidth=2.5,
             linestyle="--", label="Buy & Hold")
    ax2.plot(cum_bull.index, cum_bull.values, color="#ff9800", linewidth=2,
             label="Pure Regime (Bull=1)")
    ax2.plot(cum_bm.index, cum_bm.values, color="#e65100", linewidth=2,
             linestyle="-.", label="Regime Bull=1 Mixed=0.5")
    ax2.axhline(1.0, color="black", linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.set_ylabel("Cumulative Return", fontsize=12)
    format_date_axis(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_pure_regime.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [4/8] Pure Regime")


# ======================================================================
# Figure 5: Annual Returns
# ======================================================================

def fig5_full_period(gld_close, signal, regime):
    oos_start = "2016-01-01"
    oos_idx = gld_close.index[gld_close.index >= oos_start]
    sig = signal.reindex(oos_idx).dropna(subset=["position"])
    daily_ret = gld_close.pct_change().fillna(0).reindex(sig.index)

    pos = sig["position"]
    reg_oos = regime.reindex(sig.index)
    strat_ret = pos.shift(1).fillna(0) * daily_ret
    reg_ret = (reg_oos == "Bull").astype(float).shift(1).fillna(0) * daily_ret

    years = sorted(sig.index.year.unique())
    v2_annual, reg_annual, bh_annual = [], [], []

    for yr in years:
        yr_mask = sig.index.year == yr
        v2_annual.append(
            (1 + strat_ret[yr_mask]).cumprod().iloc[-1] - 1)
        reg_annual.append(
            (1 + reg_ret[yr_mask]).cumprod().iloc[-1] - 1)
        bh_annual.append(
            (1 + daily_ret[yr_mask]).cumprod().iloc[-1] - 1)

    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(years))
    width = 0.25

    ax.bar(x - width, [r*100 for r in v2_annual], width, color="#1565c0",
           label="V2: Adaptive Range", alpha=0.85)
    ax.bar(x, [r*100 for r in reg_annual], width, color="#ff9800",
           label="Pure Regime (Bull=1)", alpha=0.85)
    ax.bar(x + width, [r*100 for r in bh_annual], width, color="#9e9e9e",
           label="Buy & Hold", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=12)
    ax.set_ylabel("Annual Return (%)", fontsize=13)
    ax.set_title("Annual Returns: V2 vs Pure Regime vs Buy & Hold (2016-2026)",
                 fontsize=16, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=12)

    for i, (v2, reg, bh) in enumerate(
            zip(v2_annual, reg_annual, bh_annual)):
        ax.text(i - width, v2*100 + 0.5, f"{v2:+.0%}", ha="center",
                fontsize=9, fontweight="bold", color="#1565c0")
        ax.text(i, reg*100 + 0.5, f"{reg:+.0%}", ha="center",
                fontsize=9, fontweight="bold", color="#ff9800")
        ax.text(i + width, bh*100 + 0.5, f"{bh:+.0%}", ha="center",
                fontsize=9, fontweight="bold", color="#666")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_annual_returns.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [5/8] Annual Returns")


# ======================================================================
# Figure 6: DL Fair Value — LSTM vs Transformer vs EMA
# ======================================================================

def fig6_dl_fair_value(gld_close, dl_data, recent_start):
    if "lstm_fv" not in dl_data:
        print("  [6/8] SKIPPED (no DL fair value data)")
        return

    lstm = dl_data["lstm_fv"]
    has_tf = "transformer_fv" in dl_data

    # Use last 2 years for clarity
    show_start = recent_start - pd.Timedelta(days=365)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 2],
                             sharex=True)

    # Top: Price + Fair Values
    ax1 = axes[0]
    idx = lstm.index[lstm.index >= show_start]
    p = gld_close.reindex(idx)

    ax1.plot(p.index, p.values, color="#333", linewidth=2.0,
             label="GLD Close", zorder=5)

    # EMA(20) fair value
    ema20 = gld_close.ewm(span=20).mean().reindex(idx)
    ax1.plot(ema20.index, ema20.values, color="#ff9800", linewidth=1.5,
             linestyle="-.", label="EMA(20)", zorder=3, alpha=0.8)

    # LSTM fair value
    lstm_fv_col = "lstm_fair_value" if "lstm_fair_value" in lstm.columns \
        else "dl_fair_value"
    lstm_fv = lstm[lstm_fv_col].reindex(idx)
    ax1.plot(lstm_fv.index, lstm_fv.values, color="#1565c0", linewidth=1.5,
             linestyle="--", label="LSTM Fair Value", zorder=4)

    # Transformer fair value
    if has_tf:
        tf = dl_data["transformer_fv"]
        tf_fv = tf["dl_fair_value"].reindex(idx)
        ax1.plot(tf_fv.index, tf_fv.values, color="#9c27b0", linewidth=1.2,
                 linestyle=":", label="Transformer FV", zorder=3, alpha=0.7)

    ax1.legend(loc="upper left", fontsize=11)
    ax1.set_title("DL Fair Value Comparison: LSTM vs Transformer vs EMA(20)",
                  fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=13)

    # Bottom: Prediction IC (rolling 126d)
    ax2 = axes[1]
    pred_lstm = lstm["predicted_5d_return"]
    actual = lstm["actual_5d_return"]
    common_idx = pred_lstm.dropna().index.intersection(actual.dropna().index)
    common_idx = common_idx[common_idx >= show_start]

    window = 126
    rolling_ic_lstm = pd.Series(dtype=float, index=common_idx)
    for i in range(window, len(common_idx)):
        win_idx = common_idx[i-window:i]
        ic_val = stats.spearmanr(
            pred_lstm.loc[win_idx], actual.loc[win_idx])[0]
        rolling_ic_lstm.loc[common_idx[i]] = ic_val

    ax2.plot(rolling_ic_lstm.index, rolling_ic_lstm.values,
             color="#1565c0", linewidth=1.5, label="LSTM Rolling IC (126d)")

    if has_tf:
        tf = dl_data["transformer_fv"]
        pred_tf = tf["predicted_5d_return"]
        rolling_ic_tf = pd.Series(dtype=float, index=common_idx)
        for i in range(window, len(common_idx)):
            win_idx = common_idx[i-window:i]
            if win_idx[0] in pred_tf.index and win_idx[-1] in pred_tf.index:
                tf_sub = pred_tf.reindex(win_idx).dropna()
                act_sub = actual.reindex(tf_sub.index)
                if len(tf_sub) > 20:
                    rolling_ic_tf.loc[common_idx[i]] = stats.spearmanr(
                        tf_sub, act_sub)[0]
        rolling_ic_tf = rolling_ic_tf.dropna()
        ax2.plot(rolling_ic_tf.index, rolling_ic_tf.values,
                 color="#9c27b0", linewidth=1.2, linestyle=":",
                 label="Transformer Rolling IC (126d)", alpha=0.7)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0.055, color="#1565c0", linewidth=0.8, linestyle="--",
                alpha=0.5, label=f"LSTM Overall IC = +0.055")
    ax2.set_ylabel("Spearman IC", fontsize=12)
    ax2.set_ylim(-0.3, 0.4)
    ax2.legend(loc="upper left", fontsize=10)
    format_date_axis(ax2)

    # Annotate overall metrics
    lstm_ic = stats.spearmanr(pred_lstm.dropna(), actual.reindex(
        pred_lstm.dropna().index))[0]
    lstm_dir = ((pred_lstm > 0) == (actual > 0)).mean()
    txt = f"LSTM: IC={lstm_ic:+.3f}, DirAcc={lstm_dir:.1%}"
    if has_tf:
        tf_pred = dl_data["transformer_fv"]["predicted_5d_return"]
        tf_actual = dl_data["transformer_fv"]["actual_5d_return"]
        tf_ic = stats.spearmanr(tf_pred.dropna(), tf_actual.reindex(
            tf_pred.dropna().index))[0]
        tf_dir = ((tf_pred > 0) == (tf_actual > 0)).mean()
        txt += f"\nTransformer: IC={tf_ic:+.3f}, DirAcc={tf_dir:.1%}"
    ax2.annotate(txt, xy=(0.70, 0.85), xycoords="axes fraction", fontsize=11,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_dl_fair_value.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [6/8] DL Fair Value")


# ======================================================================
# Figure 7: DL Range Prediction
# ======================================================================

def fig7_dl_range(gld_close, dl_data, recent_start):
    if "range_v2" not in dl_data:
        print("  [7/8] SKIPPED (no DL range data)")
        return

    rng = dl_data["range_v2"]
    # Show last 2 years for detail
    show_start = recent_start - pd.Timedelta(days=365)
    idx = rng.index[rng.index >= show_start]
    rng_sub = rng.loc[idx]
    p = gld_close.reindex(idx)

    # Convert pct to price
    pred_upper_price = p * (1 + rng_sub["pred_upper_pct"] / 100)
    pred_lower_price = p * (1 + rng_sub["pred_lower_pct"] / 100)
    actual_upper_price = p * (1 + rng_sub["actual_upper_pct"] / 100)
    actual_lower_price = p * (1 + rng_sub["actual_lower_pct"] / 100)

    fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                             height_ratios=[3, 1.5, 1.5], sharex=True)

    # Top: Price + Predicted Range + Actual Range
    ax1 = axes[0]
    ax1.plot(p.index, p.values, color="#333", linewidth=1.8,
             label="GLD Close", zorder=5)

    ax1.fill_between(idx, pred_lower_price.values, pred_upper_price.values,
                     alpha=0.15, color="#1565c0", label="Predicted Range")
    ax1.plot(idx, pred_upper_price.values, color="#1565c0",
             linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.plot(idx, pred_lower_price.values, color="#1565c0",
             linewidth=0.8, linestyle="--", alpha=0.7)

    # Mark breaches
    upper_breach = rng_sub["actual_upper_pct"] > rng_sub["pred_upper_pct"]
    lower_breach = rng_sub["actual_lower_pct"] < rng_sub["pred_lower_pct"]
    breach_idx_u = idx[upper_breach.values]
    breach_idx_l = idx[lower_breach.values]
    if len(breach_idx_u) > 0:
        ax1.scatter(breach_idx_u,
                    actual_upper_price.loc[breach_idx_u].values,
                    color="#d32f2f", s=15, alpha=0.5, zorder=6,
                    label=f"Upper Breach (n={len(breach_idx_u)})")
    if len(breach_idx_l) > 0:
        ax1.scatter(breach_idx_l,
                    actual_lower_price.loc[breach_idx_l].values,
                    color="#2e7d32", s=15, alpha=0.5, zorder=6,
                    label=f"Lower Breach (n={len(breach_idx_l)})")

    ax1.legend(loc="upper left", fontsize=11)
    ax1.set_title("DL Range Prediction: 5-Day Upper/Lower Bounds (LSTM + Quantile Loss)",
                  fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=13)

    # Middle: Width comparison (predicted vs actual, in %)
    ax2 = axes[1]
    pred_w = rng_sub["pred_upper_pct"] - rng_sub["pred_lower_pct"]
    actual_w = rng_sub["actual_upper_pct"] - rng_sub["actual_lower_pct"]

    ax2.plot(idx, pred_w.values, color="#1565c0", linewidth=1.2,
             label=f"Predicted Width (mean={pred_w.mean():.1f}%)")
    ax2.plot(idx, actual_w.values, color="#ff9800", linewidth=1.0,
             alpha=0.7,
             label=f"Actual Width (mean={actual_w.mean():.1f}%)")
    ax2.set_ylabel("Width (%)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)

    # Bottom: Rolling coverage (126d)
    ax3 = axes[2]
    both_covered = ((rng_sub["actual_upper_pct"] <= rng_sub["pred_upper_pct"])
                    & (rng_sub["actual_lower_pct"] >= rng_sub["pred_lower_pct"]))
    rolling_cov = both_covered.rolling(126, min_periods=60).mean()

    ax3.plot(idx, rolling_cov.values, color="#4caf50", linewidth=1.8,
             label="Rolling 126d Coverage")
    ax3.axhline(both_covered.mean(), color="#4caf50", linewidth=1,
                linestyle="--", alpha=0.6,
                label=f"Overall: {both_covered.mean():.0%}")
    ax3.axhline(0.70, color="#9e9e9e", linewidth=0.8, linestyle=":",
                alpha=0.5, label="70% target")
    ax3.set_ylabel("Coverage", fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="lower left", fontsize=10)
    format_date_axis(ax3)

    # Annotate overall stats
    cov = both_covered.mean()
    tight = cov / pred_w.mean() if pred_w.mean() > 0 else 0
    ax1.annotate(
        f"Coverage: {cov:.0%}\n"
        f"Avg Width: {pred_w.mean():.1f}%\n"
        f"Width Ratio: {pred_w.mean()/actual_w.mean():.1f}x\n"
        f"Tightness: {tight:.3f}",
        xy=(0.82, 0.05), xycoords="axes fraction", fontsize=11,
        bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_dl_range.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [7/8] DL Range")


# ======================================================================
# Figure 8: All Methods Comparison Summary
# ======================================================================

def fig8_method_comparison(dl_data):
    """Bar chart comparing all prediction methods."""

    methods = []

    # Ridge baseline (from README)
    methods.append({"name": "Ridge\n(5d IC)", "ic": 0.075, "coverage": None,
                    "tightness": None, "category": "direction"})
    methods.append({"name": "XGBoost\n(5d IC)", "ic": 0.063, "coverage": None,
                    "tightness": None, "category": "direction"})

    # DL LSTM
    if "lstm_fv" in dl_data:
        lstm = dl_data["lstm_fv"]
        pred = lstm["predicted_5d_return"]
        actual = lstm["actual_5d_return"]
        ic = stats.spearmanr(pred.dropna(),
                             actual.reindex(pred.dropna().index))[0]
        methods.append({"name": "DL LSTM\n(5d IC)", "ic": ic,
                        "coverage": None, "tightness": None,
                        "category": "direction"})

    # DL Transformer
    if "transformer_fv" in dl_data:
        tf = dl_data["transformer_fv"]
        pred = tf["predicted_5d_return"]
        actual = tf["actual_5d_return"]
        ic = stats.spearmanr(pred.dropna(),
                             actual.reindex(pred.dropna().index))[0]
        methods.append({"name": "DL Transformer\n(5d IC)", "ic": ic,
                        "coverage": None, "tightness": None,
                        "category": "direction"})

    # DL Range
    if "range_v2" in dl_data:
        rng = dl_data["range_v2"]
        pu, pl = rng["pred_upper_pct"], rng["pred_lower_pct"]
        au, al = rng["actual_upper_pct"], rng["actual_lower_pct"]
        cov = ((au <= pu) & (al >= pl)).mean()
        pw = (pu - pl).mean()
        aw = (au - al).mean()
        tight = cov / pw if pw > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Direction IC comparison
    ax1 = axes[0]
    dir_methods = [m for m in methods if m["category"] == "direction"]
    names = [m["name"] for m in dir_methods]
    ics = [m["ic"] for m in dir_methods]
    colors = ["#1565c0" if ic > 0.03 else "#ff9800" if ic > 0
              else "#d32f2f" for ic in ics]
    bars = ax1.bar(names, ics, color=colors, alpha=0.85, edgecolor="white")
    for bar, ic in zip(bars, ics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{ic:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Spearman IC", fontsize=13)
    ax1.set_title("5d Return Prediction IC\n(higher = better)",
                  fontsize=14, fontweight="bold")
    ax1.set_ylim(-0.02, max(ics) * 1.4)

    # Panel 2: Range Coverage & Width
    ax2 = axes[1]
    if "range_v2" in dl_data:
        range_names = ["DL Range\nV2", "RV-based\n(baseline)"]
        coverages = [cov * 100, 63.0]  # RV-based from README
        widths = [pw, 10.2]

        x = np.arange(len(range_names))
        w = 0.35
        bars1 = ax2.bar(x - w/2, coverages, w, color="#4caf50", alpha=0.85,
                        label="Coverage (%)")
        bars2 = ax2.bar(x + w/2, widths, w, color="#ff9800", alpha=0.85,
                        label="Width (%)")

        for bar, val in zip(bars1, coverages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
        for bar, val in zip(bars2, widths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(range_names, fontsize=12)
        ax2.legend(fontsize=11)
    ax2.set_title("Range Prediction:\nCoverage vs Width",
                  fontsize=14, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=13)

    # Panel 3: Tightness (coverage / width)
    ax3 = axes[2]
    if "range_v2" in dl_data:
        tight_names = ["DL Range V2", "RV-based"]
        tightnesses = [tight, 0.063/10.2*100]  # approx for RV: 63/10.2
        # Actually tightness = coverage / width
        tightnesses = [tight, 0.63 / 10.2]

        colors3 = ["#1565c0", "#9e9e9e"]
        bars3 = ax3.bar(tight_names, tightnesses, color=colors3, alpha=0.85,
                        edgecolor="white")
        for bar, val in zip(bars3, tightnesses):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

        improvement = tightnesses[0] / tightnesses[1] if tightnesses[1] > 0 else 0
        ax3.annotate(f"DL is {improvement:.1f}x tighter",
                     xy=(0.5, 0.85), xycoords="axes fraction",
                     fontsize=13, fontweight="bold", color="#1565c0",
                     ha="center",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    ax3.set_title("Range Tightness\n(coverage / width, higher = better)",
                  fontsize=14, fontweight="bold")
    ax3.set_ylabel("Tightness", fontsize=13)

    plt.suptitle("Phase 4A: All Methods Comparison Summary",
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_method_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [8/8] Method Comparison")


# ======================================================================
# Main
# ======================================================================

def main():
    print("Phase 4A 综合报告生成...")

    features, gld_close = load_all_data()
    regime = compute_regime(features)
    dl_data = load_dl_data()

    latest = gld_close.index.max()
    recent_start = latest - pd.Timedelta(days=365)
    print(f"  近一年范围: {recent_start.date()} ~ {latest.date()}")
    print(f"  DL data loaded: {list(dl_data.keys())}")

    print("  Building V2 signal...")
    signal = build_v2_signal(features, gld_close, regime)

    print("  Generating figures...")
    fig1_regime(gld_close, regime, recent_start)
    fig2_fair_value_band(gld_close, signal, regime, recent_start)
    fig3_v2_trading(gld_close, regime, signal, recent_start)
    fig4_pure_regime(gld_close, regime, recent_start)
    fig5_full_period(gld_close, signal, regime)
    fig6_dl_fair_value(gld_close, dl_data, recent_start)
    fig7_dl_range(gld_close, dl_data, recent_start)
    fig8_method_comparison(dl_data)

    print(f"\n  8 张图已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
