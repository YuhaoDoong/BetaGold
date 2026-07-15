"""
Regime 可视化分析:
  1. 价格走势 + Regime 着色
  2. 各 Regime 期间涨跌幅分布
  3. Regime 持续时间 vs 期间收益
  4. Regime 转换前后价格表现
  5. Regime 与 RV 的交叉分析
  6. Regime 误判分析 (Bull 中下跌、Bear 中上涨)

用法:
    conda activate gold
    python src/models/analysis_regime_visual.py
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
    raw_score = regime_result["raw_score"] if "raw_score" in regime_result else None
    smoothed = regime_result["smoothed_score"] if "smoothed_score" in regime_result else None

    rv_20d = features["rv_20d"] if "rv_20d" in features.columns else None

    return gld, regime, raw_score, smoothed, rv_20d, features


def identify_segments(regime):
    """识别连续 regime 段。"""
    segments = []
    current = regime.iloc[0]
    start = regime.index[0]

    for i in range(1, len(regime)):
        if regime.iloc[i] != current:
            segments.append({
                "regime": current,
                "start": start,
                "end": regime.index[i - 1],
                "days": i - regime.index.get_loc(start),
            })
            current = regime.iloc[i]
            start = regime.index[i]

    segments.append({
        "regime": current,
        "start": start,
        "end": regime.index[-1],
        "days": len(regime) - regime.index.get_loc(start),
    })

    return pd.DataFrame(segments)


# ============================================================
# Fig 1: 价格走势 + Regime 着色 (全时段)
# ============================================================

def fig1_price_with_regime(gld, regime):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)

    ax1 = axes[0]
    common = gld.index.intersection(regime.index)
    close = gld["Close"].reindex(common)
    reg = regime.reindex(common)

    ax1.plot(common, close, color="black", linewidth=0.8, alpha=0.8)

    # 着色 regime 区域
    for r, color in COLORS.items():
        mask = reg == r
        starts = mask & (~mask.shift(1, fill_value=False))
        ends = mask & (~mask.shift(-1, fill_value=False))
        for s, e in zip(common[starts], common[ends]):
            ax1.axvspan(s, e, alpha=0.2, color=color)

    ax1.set_ylabel("GLD Price ($)")
    ax1.set_title("GLD Price with Regime Classification")
    ax1.legend(handles=[Patch(color=c, alpha=0.3, label=r) for r, c in COLORS.items()],
               loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 下面画 regime score
    ax2 = axes[1]
    segments = identify_segments(reg)
    for _, seg in segments.iterrows():
        mask = (common >= seg["start"]) & (common <= seg["end"])
        seg_close = close[mask]
        if len(seg_close) < 2:
            continue
        ret = (seg_close.iloc[-1] / seg_close.iloc[0] - 1) * 100
        mid = seg["start"] + (seg["end"] - seg["start"]) / 2
        ax2.bar(mid, ret, width=(seg["end"] - seg["start"]).days,
                color=COLORS[seg["regime"]], alpha=0.7, edgecolor="gray", linewidth=0.5)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Segment Return (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_price_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Fig 2: 各 Regime 前瞻收益分布 (boxplot + violin)
# ============================================================

def fig2_return_distributions(gld, regime):
    close = gld["Close"]
    common = close.index.intersection(regime.index)
    close = close.reindex(common)
    reg = regime.reindex(common)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, horizon in enumerate([5, 10, 20]):
        ax = axes[i]
        fwd = (close.shift(-horizon) / close - 1) * 100
        fwd = fwd.reindex(common).dropna()

        data = []
        labels = []
        for r in ["Bull", "Mixed", "Bear"]:
            mask = reg.reindex(fwd.index) == r
            vals = fwd[mask].dropna()
            if len(vals) > 10:
                data.append(vals.values)
                labels.append(f"{r}\n(n={len(vals)})")

        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            color = list(COLORS.values())[j]
            pc.set_facecolor(color)
            pc.set_alpha(0.4)

        # overlay box
        bp = ax.boxplot(data, widths=0.15, patch_artist=False,
                        medianprops=dict(color="red", linewidth=2),
                        whiskerprops=dict(linewidth=0.5),
                        flierprops=dict(markersize=2, alpha=0.3))

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(f"{horizon}d Forward Return Distribution")
        ax.set_ylabel("Return (%)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

        # 标注均值
        for j, vals in enumerate(data):
            mean_val = np.mean(vals)
            ax.annotate(f"μ={mean_val:+.2f}%",
                        xy=(j + 1, mean_val),
                        fontsize=9, fontweight="bold", color="darkblue",
                        ha="center", va="bottom")

    plt.suptitle("Forward Return Distribution by Regime", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_return_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Fig 3: Regime 段落统计 (持续天数 vs 段内收益)
# ============================================================

def fig3_segment_stats(gld, regime):
    close = gld["Close"]
    common = close.index.intersection(regime.index)
    close = close.reindex(common)
    reg = regime.reindex(common)

    segments = identify_segments(reg)

    # 计算段内收益
    for i, seg in segments.iterrows():
        mask = (common >= seg["start"]) & (common <= seg["end"])
        seg_close = close[mask]
        if len(seg_close) >= 2:
            segments.loc[i, "return_pct"] = (seg_close.iloc[-1] / seg_close.iloc[0] - 1) * 100
            segments.loc[i, "ann_return"] = segments.loc[i, "return_pct"] / seg["days"] * 252
        else:
            segments.loc[i, "return_pct"] = 0
            segments.loc[i, "ann_return"] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 散点图: 天数 vs 收益
    ax1 = axes[0]
    for r in ["Bull", "Mixed", "Bear"]:
        mask = segments["regime"] == r
        sub = segments[mask]
        ax1.scatter(sub["days"], sub["return_pct"],
                    c=COLORS[r], label=f"{r} (n={len(sub)})",
                    s=80, alpha=0.7, edgecolors="gray")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Segment Duration (days)")
    ax1.set_ylabel("Segment Return (%)")
    ax1.set_title("Duration vs Return by Regime")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 持续天数分布
    ax2 = axes[1]
    for r in ["Bull", "Mixed", "Bear"]:
        mask = segments["regime"] == r
        sub = segments[mask]
        if len(sub) > 0:
            ax2.hist(sub["days"], bins=15, alpha=0.5, color=COLORS[r],
                     label=f"{r} (med={sub['days'].median():.0f}d)", edgecolor="gray")
    ax2.set_xlabel("Duration (days)")
    ax2.set_ylabel("Count")
    ax2.set_title("Regime Segment Duration Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 段内年化收益分布
    ax3 = axes[2]
    for r in ["Bull", "Mixed", "Bear"]:
        mask = (segments["regime"] == r) & (segments["days"] >= 20)
        sub = segments[mask]
        if len(sub) >= 3:
            ax3.hist(sub["ann_return"].clip(-100, 100), bins=15, alpha=0.5,
                     color=COLORS[r],
                     label=f"{r} (μ={sub['ann_return'].mean():+.0f}%)",
                     edgecolor="gray")
    ax3.set_xlabel("Annualized Return (%)")
    ax3.set_ylabel("Count")
    ax3.set_title("Annualized Return per Segment (≥20d)")
    ax3.legend()
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Regime Segment Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_segment_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 打印段落表
    print(f"\n  === Regime 段落明细 ===")
    print(f"  {'Regime':8s} {'Start':>12s} {'End':>12s} {'Days':>6s} {'Return':>8s} {'AnnRet':>8s}")
    print(f"  {'-' * 58}")
    for _, seg in segments.iterrows():
        print(f"  {seg['regime']:8s} {seg['start'].strftime('%Y-%m-%d'):>12s} "
              f"{seg['end'].strftime('%Y-%m-%d'):>12s} {seg['days']:6.0f} "
              f"{seg['return_pct']:+8.1f}% {seg['ann_return']:+8.0f}%")

    # 汇总
    print(f"\n  === 汇总 ===")
    for r in ["Bull", "Mixed", "Bear"]:
        mask = segments["regime"] == r
        sub = segments[mask]
        if len(sub) == 0:
            continue
        total_days = sub["days"].sum()
        avg_ret = sub["return_pct"].mean()
        win = (sub["return_pct"] > 0).mean()
        print(f"  {r:8s}: {len(sub):3d} 段, 总{total_days:5.0f}天, "
              f"段均收益={avg_ret:+.1f}%, 正收益率={win:.0%}, "
              f"段均天数={sub['days'].mean():.0f}d")


# ============================================================
# Fig 4: Regime 转换效应
# ============================================================

def fig4_transition_effect(gld, regime):
    close = gld["Close"]
    common = close.index.intersection(regime.index)
    close_s = close.reindex(common)
    reg = regime.reindex(common)

    # 找转换点
    transitions = []
    for i in range(1, len(reg)):
        if reg.iloc[i] != reg.iloc[i - 1]:
            transitions.append({
                "date": reg.index[i],
                "from": reg.iloc[i - 1],
                "to": reg.iloc[i],
                "type": f"{reg.iloc[i-1]}→{reg.iloc[i]}"
            })
    trans_df = pd.DataFrame(transitions)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 关键转换: Bull进入/退出
    key_transitions = [
        ("→Bull (进入)", lambda r: r["to"] == "Bull"),
        ("Bull→ (退出)", lambda r: r["from"] == "Bull"),
        ("→Bear (进入)", lambda r: r["to"] == "Bear"),
        ("Bear→ (退出)", lambda r: r["from"] == "Bear"),
    ]

    for idx, (title, filt) in enumerate(key_transitions):
        ax = axes[idx // 2][idx % 2]
        mask = trans_df.apply(filt, axis=1)
        trans_dates = trans_df[mask]["date"].values

        if len(trans_dates) < 2:
            ax.set_title(f"{title} (n={len(trans_dates)}, 样本不足)")
            continue

        # 叠加前后 40 天收益路径
        window = 40
        paths = []
        for td in trans_dates:
            td = pd.Timestamp(td)
            loc = close_s.index.get_indexer([td], method="nearest")[0]
            start = max(0, loc - window)
            end = min(len(close_s), loc + window + 1)
            segment = close_s.iloc[start:end]
            base = close_s.iloc[loc]
            ret_path = (segment / base - 1) * 100
            # 对齐到 day 0
            day_idx = np.arange(-(loc - start), end - loc)
            path_s = pd.Series(ret_path.values, index=day_idx)
            paths.append(path_s)
            ax.plot(day_idx, ret_path.values, color="gray", alpha=0.2, linewidth=0.8)

        # 均值路径
        all_days = pd.DataFrame(paths).T
        mean_path = all_days.mean(axis=1)
        ax.plot(mean_path.index, mean_path.values, color="red",
                linewidth=2.5, label=f"Mean (n={len(trans_dates)})")

        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Trading Days from Transition")
        ax.set_ylabel("Return (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Price Path Around Regime Transitions", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_transition_effect.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 打印转换统计
    print(f"\n  === 转换统计 ===")
    for ttype in trans_df["type"].unique():
        mask = trans_df["type"] == ttype
        dates = trans_df[mask]["date"]
        n = len(dates)

        fwd_5d = []
        fwd_10d = []
        fwd_20d = []
        for d in dates:
            d = pd.Timestamp(d)
            loc = close_s.index.get_indexer([d], method="nearest")[0]
            if loc + 20 < len(close_s):
                p0 = close_s.iloc[loc]
                fwd_5d.append((close_s.iloc[loc + 5] / p0 - 1) * 100)
                fwd_10d.append((close_s.iloc[loc + 10] / p0 - 1) * 100)
                fwd_20d.append((close_s.iloc[loc + 20] / p0 - 1) * 100)

        if len(fwd_5d) > 0:
            print(f"  {ttype:20s} (n={n:2d}): "
                  f"5d={np.mean(fwd_5d):+.2f}% (w={np.mean(np.array(fwd_5d)>0):.0%}), "
                  f"10d={np.mean(fwd_10d):+.2f}% (w={np.mean(np.array(fwd_10d)>0):.0%}), "
                  f"20d={np.mean(fwd_20d):+.2f}% (w={np.mean(np.array(fwd_20d)>0):.0%})")


# ============================================================
# Fig 5: Regime 误判分析 (Bull中跌, Bear中涨)
# ============================================================

def fig5_misclassification(gld, regime):
    close = gld["Close"]
    common = close.index.intersection(regime.index)
    close_s = close.reindex(common)
    reg = regime.reindex(common)

    fwd_5d = (close_s.shift(-5) / close_s - 1) * 100
    fwd_10d = (close_s.shift(-10) / close_s - 1) * 100
    fwd_20d = (close_s.shift(-20) / close_s - 1) * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bull 中下跌天数占比 (rolling)
    ax1 = axes[0][0]
    bull_mask = reg == "Bull"
    bull_down_5d = (fwd_5d < 0) & bull_mask
    # rolling 胜率 (60天窗口)
    bull_win_rate = pd.Series(np.nan, index=common)
    for d in common:
        window = bull_mask.loc[:d].tail(60)
        if window.sum() >= 10:
            win = (fwd_5d.reindex(window[window].index) > 0).mean()
            bull_win_rate.loc[d] = win

    bull_win_rate = bull_win_rate.dropna()
    ax1.plot(bull_win_rate.index, bull_win_rate.values, color=COLORS["Bull"], linewidth=1)
    ax1.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax1.axhline(0.6, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.fill_between(bull_win_rate.index, 0.5, bull_win_rate.values,
                     where=bull_win_rate.values > 0.5,
                     color=COLORS["Bull"], alpha=0.2)
    ax1.fill_between(bull_win_rate.index, 0.5, bull_win_rate.values,
                     where=bull_win_rate.values < 0.5,
                     color=COLORS["Bear"], alpha=0.2)
    ax1.set_ylabel("5d Win Rate (rolling 60d)")
    ax1.set_title("Bull Regime 5d Win Rate Over Time")
    ax1.set_ylim(0.2, 0.9)
    ax1.grid(True, alpha=0.3)

    # 2. 年度 Regime 胜率
    ax2 = axes[0][1]
    years = common.year
    year_stats = []
    for yr in sorted(set(years)):
        yr_mask = years == yr
        for r in ["Bull", "Mixed", "Bear"]:
            r_mask = (reg == r) & yr_mask
            if r_mask.sum() >= 10:
                wr_5 = (fwd_5d[r_mask] > 0).mean()
                wr_10 = (fwd_10d[r_mask] > 0).mean()
                avg_ret = fwd_5d[r_mask].mean()
                year_stats.append({
                    "year": yr, "regime": r,
                    "n": r_mask.sum(),
                    "wr_5d": wr_5, "wr_10d": wr_10,
                    "avg_5d": avg_ret,
                })

    ys_df = pd.DataFrame(year_stats)
    width = 0.25
    for i, r in enumerate(["Bull", "Mixed", "Bear"]):
        sub = ys_df[ys_df["regime"] == r]
        if len(sub) > 0:
            ax2.bar(sub["year"] + (i - 1) * width, sub["wr_5d"],
                    width=width, color=COLORS[r], alpha=0.7, label=r)
    ax2.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("5d Win Rate")
    ax2.set_title("5d Win Rate by Regime and Year")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Bull 假阳性: Bull中5d收益为负的天数占比, 按年
    ax3 = axes[1][0]
    bull_year = ys_df[ys_df["regime"] == "Bull"].copy()
    if len(bull_year) > 0:
        bars = ax3.bar(bull_year["year"], bull_year["avg_5d"],
                       color=[COLORS["Bull"] if v > 0 else COLORS["Bear"]
                              for v in bull_year["avg_5d"]],
                       alpha=0.7, edgecolor="gray")
        ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        # 标注样本量
        for _, row in bull_year.iterrows():
            ax3.annotate(f"n={row['n']:.0f}", xy=(row["year"], 0),
                         fontsize=7, ha="center", va="bottom", rotation=90)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Avg 5d Return (%)")
        ax3.set_title("Bull Regime: Average 5d Return by Year")
        ax3.grid(True, alpha=0.3, axis="y")

    # 4. Regime 分类准确度: 用20d收益方向作为ground truth
    ax4 = axes[1][1]
    # "正确分类": Bull且20d涨, Bear且20d跌, Mixed不评判
    bull_correct = ((reg == "Bull") & (fwd_20d > 0)).sum()
    bull_total = (reg == "Bull").sum()
    bear_correct = ((reg == "Bear") & (fwd_20d < 0)).sum()
    bear_total = (reg == "Bear").sum()
    # 也看 Bull 的 "严重错误": 20d跌超过2%
    bull_severe = ((reg == "Bull") & (fwd_20d < -2)).sum()
    bear_severe = ((reg == "Bear") & (fwd_20d > 2)).sum()

    categories = ["Bull\n20d涨", "Bull\n20d跌", "Bull\n20d跌>2%",
                   "Bear\n20d跌", "Bear\n20d涨", "Bear\n20d涨>2%"]
    values = [
        bull_correct, bull_total - bull_correct, bull_severe,
        bear_correct, bear_total - bear_correct, bear_severe,
    ]
    colors_bar = [COLORS["Bull"], "#ffcccc", COLORS["Bear"],
                  COLORS["Bear"], "#ccffcc", COLORS["Bull"]]

    ax4.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor="gray")
    # 标注百分比
    for j, (cat, val) in enumerate(zip(categories, values)):
        total = bull_total if j < 3 else bear_total
        pct = val / total * 100 if total > 0 else 0
        ax4.annotate(f"{val}\n({pct:.0f}%)", xy=(j, val),
                     fontsize=9, ha="center", va="bottom")

    ax4.set_title(f"Regime Classification Accuracy (20d ground truth)")
    ax4.set_ylabel("Count (days)")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Regime Misclassification Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_misclassification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 打印详细统计
    print(f"\n  === Regime 分类准确度 ===")
    print(f"  Bull: {bull_total}天, 20d涨={bull_correct} ({bull_correct/bull_total:.1%}), "
          f"20d跌={bull_total-bull_correct} ({(bull_total-bull_correct)/bull_total:.1%}), "
          f"严重误判(跌>2%)={bull_severe} ({bull_severe/bull_total:.1%})")
    if bear_total > 0:
        print(f"  Bear: {bear_total}天, 20d跌={bear_correct} ({bear_correct/bear_total:.1%}), "
              f"20d涨={bear_total-bear_correct} ({(bear_total-bear_correct)/bear_total:.1%}), "
              f"严重误判(涨>2%)={bear_severe} ({bear_severe/bear_total:.1%})")


# ============================================================
# Fig 6: Regime 与 RV 交叉分析
# ============================================================

def fig6_regime_rv_cross(regime, rv_20d, gld):
    if rv_20d is None:
        print("  RV data not available, skipping fig6")
        return

    close = gld["Close"]
    common = regime.index.intersection(rv_20d.dropna().index).intersection(close.index)
    reg = regime.reindex(common)
    rv = rv_20d.reindex(common)
    close_s = close.reindex(common)

    fwd_5d = (close_s.shift(-5) / close_s - 1) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. RV 分布 by Regime
    ax1 = axes[0]
    for r in ["Bull", "Mixed", "Bear"]:
        mask = reg == r
        if mask.sum() > 10:
            ax1.hist(rv[mask], bins=30, alpha=0.4, color=COLORS[r],
                     label=f"{r} (μ={rv[mask].mean():.1f})", edgecolor="gray", density=True)
    ax1.set_xlabel("RV 20d")
    ax1.set_ylabel("Density")
    ax1.set_title("RV Distribution by Regime")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Regime × RV 分位 → 5d胜率
    ax2 = axes[1]
    rv_q = pd.qcut(rv, 3, labels=["Low", "Med", "High"])

    matrix = pd.DataFrame(index=["Bull", "Mixed", "Bear"], columns=["Low", "Med", "High"])
    for r in ["Bull", "Mixed", "Bear"]:
        for q in ["Low", "Med", "High"]:
            mask = (reg == r) & (rv_q == q)
            if mask.sum() >= 20:
                matrix.loc[r, q] = (fwd_5d[mask] > 0).mean()
            else:
                matrix.loc[r, q] = np.nan

    matrix_float = matrix.astype(float)
    im = ax2.imshow(matrix_float.values, cmap="RdYlGn", vmin=0.35, vmax=0.70, aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Low RV", "Med RV", "High RV"])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(["Bull", "Mixed", "Bear"])
    ax2.set_title("5d Win Rate: Regime × RV")

    for i in range(3):
        for j in range(3):
            val = matrix_float.iloc[i, j]
            if not np.isnan(val):
                mask = (reg == ["Bull", "Mixed", "Bear"][i]) & (rv_q == ["Low", "Med", "High"][j])
                n = mask.sum()
                ax2.text(j, i, f"{val:.0%}\n(n={n})", ha="center", va="center",
                         fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax2)

    # 3. Regime × RV → 5d 平均收益
    ax3 = axes[2]
    matrix2 = pd.DataFrame(index=["Bull", "Mixed", "Bear"], columns=["Low", "Med", "High"])
    for r in ["Bull", "Mixed", "Bear"]:
        for q in ["Low", "Med", "High"]:
            mask = (reg == r) & (rv_q == q)
            if mask.sum() >= 20:
                matrix2.loc[r, q] = fwd_5d[mask].mean()
            else:
                matrix2.loc[r, q] = np.nan

    matrix2_float = matrix2.astype(float)
    im2 = ax3.imshow(matrix2_float.values, cmap="RdYlGn", vmin=-0.5, vmax=0.8, aspect="auto")
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["Low RV", "Med RV", "High RV"])
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(["Bull", "Mixed", "Bear"])
    ax3.set_title("5d Mean Return: Regime × RV")

    for i in range(3):
        for j in range(3):
            val = matrix2_float.iloc[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f"{val:+.2f}%", ha="center", va="center",
                         fontsize=10, fontweight="bold")

    plt.colorbar(im2, ax=ax3)

    plt.suptitle("Regime × Volatility Cross Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_regime_rv_cross.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Regime 可视化分析")
    print("=" * 70)

    gld, regime, raw_score, smoothed, rv_20d, features = load_all()

    print(f"\n  数据: {regime.index[0].date()} ~ {regime.index[-1].date()}")
    print(f"  Bull: {(regime=='Bull').sum()}d ({(regime=='Bull').mean():.1%})")
    print(f"  Mixed: {(regime=='Mixed').sum()}d ({(regime=='Mixed').mean():.1%})")
    print(f"  Bear: {(regime=='Bear').sum()}d ({(regime=='Bear').mean():.1%})")

    fig1_price_with_regime(gld, regime)
    fig2_return_distributions(gld, regime)
    fig3_segment_stats(gld, regime)
    fig4_transition_effect(gld, regime)
    fig5_misclassification(gld, regime)
    fig6_regime_rv_cross(regime, rv_20d, gld)

    print(f"\n  所有图表保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
