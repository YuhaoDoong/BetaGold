"""
近几年 Regime 可视化 + RV 均值回归分析

用法:
    conda activate gold
    python src/models/analysis_regime_rv_recent.py
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
from matplotlib.patches import Patch
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "regime_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"Bull": "#2ecc71", "Mixed": "#f39c12", "Bear": "#e74c3c"}


def load_all():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime_result = RegimeClassifier().classify(features[feat_cols])
    regime = regime_result["regime"]
    smoothed = regime_result.get("smoothed_score", None)

    rv_20d = features["rv_20d"] if "rv_20d" in features.columns else None
    hv_5d = features["hv_5d"] if "hv_5d" in features.columns else None

    # GVZ
    gvz_path = os.path.join(raw_dir, "volatility", "gvz.csv")
    gvz_df = pd.read_csv(gvz_path, index_col=0, parse_dates=True)
    gvz = gvz_df.iloc[:, 0]

    return gld, regime, smoothed, rv_20d, hv_5d, gvz, features


# ============================================================
# Fig 7: 近几年 Regime + 价格 (2022-2025, 细节)
# ============================================================

def fig7_recent_regime(gld, regime, smoothed):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                             gridspec_kw={"height_ratios": [3, 1.5, 1]},
                             sharex=True)

    # 近4年
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2026-03-31")

    common = gld.index.intersection(regime.index)
    mask = (common >= start) & (common <= end)
    common = common[mask]
    close = gld["Close"].reindex(common)
    reg = regime.reindex(common)

    # --- Panel 1: 价格 + Regime 着色 ---
    ax1 = axes[0]
    ax1.plot(common, close, color="black", linewidth=1.2)

    for r, color in COLORS.items():
        rmask = reg == r
        starts = rmask & (~rmask.shift(1, fill_value=False))
        ends = rmask & (~rmask.shift(-1, fill_value=False))
        for s, e in zip(common[starts], common[ends]):
            ax1.axvspan(s, e, alpha=0.25, color=color)

    # 标注关键事件
    events = [
        ("2022-02-24", "俄乌战争", "top"),
        ("2022-11-01", "加息见顶预期", "bottom"),
        ("2023-10-07", "巴以冲突", "bottom"),
        ("2024-03-28", "Bull开始", "bottom"),
        ("2025-01-20", "Trump就职", "top"),
    ]
    for date_str, label, pos in events:
        d = pd.Timestamp(date_str)
        if d in close.index:
            y = close.loc[d]
        else:
            nearest = close.index[close.index.get_indexer([d], method="nearest")[0]]
            y = close.loc[nearest]
            d = nearest
        va = "bottom" if pos == "bottom" else "top"
        offset = 8 if pos == "bottom" else -8
        ax1.annotate(label, xy=(d, y), fontsize=8,
                     xytext=(0, offset), textcoords="offset points",
                     ha="center", va=va,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax1.set_ylabel("GLD Price ($)")
    ax1.set_title("GLD Price with Regime (2022-2025)")
    ax1.legend(handles=[Patch(color=c, alpha=0.3, label=r) for r, c in COLORS.items()],
               loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: 5d 前瞻收益 + Regime ---
    ax2 = axes[1]
    fwd_5d = (close.shift(-5) / close - 1) * 100
    for r, color in COLORS.items():
        rmask = reg == r
        if rmask.sum() > 0:
            ax2.bar(common[rmask], fwd_5d[rmask], color=color, alpha=0.5, width=2)

    ax2.axhline(0, color="black", linewidth=0.5)

    # rolling 20d 均值
    rolling_mean = fwd_5d.rolling(20).mean()
    ax2.plot(common, rolling_mean, color="navy", linewidth=1.5, label="20d MA of 5d fwd return")
    ax2.set_ylabel("5d Fwd Return (%)")
    ax2.legend(loc="upper left")
    ax2.set_ylim(-8, 8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Regime Score ---
    ax3 = axes[2]
    if smoothed is not None:
        sm = smoothed.reindex(common)
        ax3.plot(common, sm, color="purple", linewidth=1.2)
        ax3.axhline(2, color=COLORS["Bull"], linewidth=1, linestyle="--", alpha=0.5, label="Bull thresh")
        ax3.axhline(-2, color=COLORS["Bear"], linewidth=1, linestyle="--", alpha=0.5, label="Bear thresh")
        ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax3.fill_between(common, sm, 2, where=sm > 2, color=COLORS["Bull"], alpha=0.2)
        ax3.fill_between(common, sm, -2, where=sm < -2, color=COLORS["Bear"], alpha=0.2)
        ax3.set_ylabel("Smoothed Score")
        ax3.legend(loc="upper left", fontsize=8)
    else:
        # 画一个简单的 regime 时间线
        regime_num = reg.map({"Bull": 1, "Mixed": 0, "Bear": -1})
        ax3.fill_between(common, 0, regime_num, where=regime_num > 0,
                         color=COLORS["Bull"], alpha=0.5)
        ax3.fill_between(common, 0, regime_num, where=regime_num < 0,
                         color=COLORS["Bear"], alpha=0.5)
        ax3.set_ylabel("Regime")
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(["Bear", "Mixed", "Bull"])

    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_recent_regime_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 打印近几年统计
    print(f"\n  === 近几年 Regime 统计 (2022-2025) ===")
    fwd_10d = (close.shift(-10) / close - 1) * 100
    fwd_20d = (close.shift(-20) / close - 1) * 100

    for yr in [2022, 2023, 2024, 2025]:
        yr_mask = common.year == yr
        if yr_mask.sum() == 0:
            continue
        print(f"\n  --- {yr} ---")
        for r in ["Bull", "Mixed", "Bear"]:
            rmask = (reg == r) & yr_mask
            n = rmask.sum()
            if n < 5:
                if n > 0:
                    print(f"  {r:8s}: {n:3d}天 ({n/yr_mask.sum():.0%}) — 样本不足")
                continue
            wr5 = (fwd_5d[rmask] > 0).mean()
            avg5 = fwd_5d[rmask].mean()
            wr10 = (fwd_10d[rmask] > 0).mean()
            avg10 = fwd_10d[rmask].mean()
            wr20 = (fwd_20d[rmask].dropna() > 0).mean() if fwd_20d[rmask].dropna().shape[0] > 0 else np.nan
            avg20 = fwd_20d[rmask].mean()
            print(f"  {r:8s}: {n:3d}天 ({n/yr_mask.sum():.0%}) | "
                  f"5d: {wr5:.0%} ({avg5:+.2f}%) | "
                  f"10d: {wr10:.0%} ({avg10:+.2f}%) | "
                  f"20d: {wr20:.0%} ({avg20:+.2f}%)")


# ============================================================
# Fig 8: RV 均值回归分析
# ============================================================

def fig8_rv_mean_reversion(rv_20d, gvz, gld):
    if rv_20d is None:
        print("  RV data not available, skipping")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    rv = rv_20d.dropna()

    # --- Panel 1: RV 历史 + 均值 ---
    ax = axes[0][0]
    ax.plot(rv.index, rv.values, color="steelblue", linewidth=0.8, alpha=0.8)
    mean_rv = rv.mean()
    ax.axhline(mean_rv, color="red", linewidth=1, linestyle="--",
               label=f"Mean={mean_rv:.1f}")
    # 分位数
    q25 = rv.quantile(0.25)
    q75 = rv.quantile(0.75)
    ax.axhline(q25, color="green", linewidth=0.5, linestyle=":", label=f"Q25={q25:.1f}")
    ax.axhline(q75, color="orange", linewidth=0.5, linestyle=":", label=f"Q75={q75:.1f}")
    ax.set_title("RV 20d Historical")
    ax.set_ylabel("RV 20d")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: RV 自相关 (ACF) ---
    ax = axes[0][1]
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(rv.values, nlags=60, fft=True)
    ax.bar(range(61), acf_vals, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.96 / np.sqrt(len(rv)), color="red", linewidth=0.5, linestyle="--")
    ax.axhline(-1.96 / np.sqrt(len(rv)), color="red", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("RV 20d Autocorrelation")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: RV 变化 → 未来 RV 变化 (均值回归) ---
    ax = axes[0][2]
    rv_change = rv.pct_change(20) * 100  # 过去20天RV变化%
    rv_fwd_change = rv.pct_change(20).shift(-20) * 100  # 未来20天RV变化%
    valid = rv_change.dropna().index.intersection(rv_fwd_change.dropna().index)
    x = rv_change.reindex(valid).values
    y = rv_fwd_change.reindex(valid).values

    ax.scatter(x, y, alpha=0.1, s=5, color="steelblue")

    # 分桶均值
    bins = np.percentile(x, np.arange(0, 101, 10))
    bin_centers = []
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(y[mask].mean())
    ax.plot(bin_centers, bin_means, "r-o", linewidth=2, markersize=6, label="Bin mean")

    rho, p = stats.spearmanr(x, y)
    ax.set_xlabel("RV Change Past 20d (%)")
    ax.set_ylabel("RV Change Next 20d (%)")
    ax.set_title(f"RV Mean Reversion (ρ={rho:.2f}, p={p:.1e})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: RV 分位 → 未来 RV 方向 ---
    ax = axes[1][0]
    rv_pctile = rv.rolling(252).rank(pct=True)
    rv_fwd_dir = (rv.shift(-20) < rv).astype(int)  # 1=未来RV下降

    valid2 = rv_pctile.dropna().index.intersection(rv_fwd_dir.dropna().index)
    pctile_vals = rv_pctile.reindex(valid2)
    fwd_dir = rv_fwd_dir.reindex(valid2)

    # 分桶
    pctile_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pctile_labels = ["0-10", "10-20", "20-30", "30-40", "40-50",
                     "50-60", "60-70", "70-80", "80-90", "90-100"]
    bucket = pd.cut(pctile_vals, bins=pctile_bins, labels=pctile_labels)

    decline_rates = []
    counts = []
    for lab in pctile_labels:
        mask = bucket == lab
        if mask.sum() > 20:
            decline_rates.append(fwd_dir[mask].mean())
            counts.append(mask.sum())
        else:
            decline_rates.append(np.nan)
            counts.append(0)

    colors = ["#e74c3c" if r > 0.55 else "#2ecc71" if r < 0.45 else "#f39c12"
              for r in decline_rates]
    bars = ax.bar(pctile_labels, decline_rates, color=colors, alpha=0.7, edgecolor="gray")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("RV Percentile (252d)")
    ax.set_ylabel("P(RV declines in 20d)")
    ax.set_title("RV Mean Reversion by Percentile")
    ax.set_ylim(0.3, 0.7)

    for i, (rate, n) in enumerate(zip(decline_rates, counts)):
        if not np.isnan(rate):
            ax.annotate(f"{rate:.0%}\nn={n}", xy=(i, rate),
                        fontsize=7, ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 5: 近1年 RV + GVZ + 价格 ---
    ax = axes[1][1]
    start_1y = pd.Timestamp("2025-03-09") - pd.DateOffset(years=1)
    end_1y = pd.Timestamp("2025-03-09")

    rv_1y = rv.loc[start_1y:end_1y]
    close_1y = gld["Close"].loc[start_1y:end_1y]

    ax_price = ax.twinx()
    ax_price.plot(close_1y.index, close_1y.values, color="black",
                  linewidth=1.2, alpha=0.6, label="GLD Price")
    ax_price.set_ylabel("GLD Price ($)", color="black")

    ax.plot(rv_1y.index, rv_1y.values, color="steelblue", linewidth=1.5, label="RV 20d")
    ax.axhline(rv_1y.mean(), color="steelblue", linewidth=0.5, linestyle="--")

    # GVZ overlay
    gvz_1y = gvz.loc[start_1y:end_1y] if len(gvz.loc[start_1y:end_1y]) > 0 else None
    if gvz_1y is not None and len(gvz_1y) > 0:
        ax.plot(gvz_1y.index, gvz_1y.values, color="darkorange",
                linewidth=1.5, alpha=0.8, label="GVZ")

    ax.set_ylabel("Volatility")
    ax.set_title("Recent 1 Year: RV + GVZ + Price")
    ax.legend(loc="upper left", fontsize=8)
    ax_price.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Panel 6: 近1年 RV 变化与价格变化的关系 ---
    ax = axes[1][2]
    rv_1y_full = rv.loc[start_1y:end_1y]
    close_1y_full = gld["Close"].reindex(rv_1y_full.index)

    rv_5d_change = rv_1y_full.pct_change(5) * 100
    price_5d_fwd = (close_1y_full.shift(-5) / close_1y_full - 1) * 100

    valid3 = rv_5d_change.dropna().index.intersection(price_5d_fwd.dropna().index)
    x3 = rv_5d_change.reindex(valid3)
    y3 = price_5d_fwd.reindex(valid3)

    ax.scatter(x3, y3, alpha=0.5, s=20, color="steelblue")

    # 分桶
    bins3 = np.percentile(x3, [0, 20, 40, 60, 80, 100])
    for i in range(len(bins3) - 1):
        mask = (x3 >= bins3[i]) & (x3 < bins3[i + 1])
        if mask.sum() > 5:
            mid = (bins3[i] + bins3[i + 1]) / 2
            avg = y3[mask].mean()
            ax.plot(mid, avg, "ro", markersize=10)
            ax.annotate(f"{avg:+.2f}%", xy=(mid, avg),
                        fontsize=9, ha="center", va="bottom", color="red")

    rho3, p3 = stats.spearmanr(x3, y3)
    ax.set_xlabel("RV 5d Change (%)")
    ax.set_ylabel("Price 5d Forward Return (%)")
    ax.set_title(f"Recent 1Y: RV Change → Price (ρ={rho3:.2f})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)

    plt.suptitle("RV Mean Reversion & Recent Volatility Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "08_rv_mean_reversion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 打印统计
    print(f"\n  === RV 均值回归统计 ===")
    print(f"  全样本: n={len(rv)}, mean={rv.mean():.1f}, std={rv.std():.1f}")
    print(f"  Spearman(RV_change_20d, RV_fwd_change_20d): rho={rho:.3f}, p={p:.1e}")
    if rho < -0.2 and p < 0.01:
        print(f"  → RV 有显著均值回归特征 (过去涨→未来倾向跌)")
    elif rho < 0:
        print(f"  → RV 有弱均值回归")
    else:
        print(f"  → RV 无明显均值回归")

    # 按分位数的均值回归强度
    print(f"\n  === RV 分位 → 未来20d RV 下降概率 ===")
    for lab, rate, n in zip(pctile_labels, decline_rates, counts):
        if not np.isnan(rate) and n > 0:
            direction = "↓回落" if rate > 0.55 else "↑扩大" if rate < 0.45 else "→持平"
            print(f"  {lab:>8s}: P(decline)={rate:.0%} (n={n:4d}) {direction}")

    # 近1年RV统计
    print(f"\n  === 近1年 RV 统计 ===")
    print(f"  均值: {rv_1y.mean():.1f}")
    print(f"  最小: {rv_1y.min():.1f} ({rv_1y.idxmin().strftime('%Y-%m-%d')})")
    print(f"  最大: {rv_1y.max():.1f} ({rv_1y.idxmax().strftime('%Y-%m-%d')})")
    print(f"  当前: {rv_1y.iloc[-1]:.1f}")
    pctile_now = (rv < rv_1y.iloc[-1]).mean()
    print(f"  当前分位: {pctile_now:.0%} (历史)")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  近几年 Regime + RV 分析")
    print("=" * 70)

    gld, regime, smoothed, rv_20d, hv_5d, gvz, features = load_all()

    fig7_recent_regime(gld, regime, smoothed)
    fig8_rv_mean_reversion(rv_20d, gvz, gld)

    print(f"\n  所有图表保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
