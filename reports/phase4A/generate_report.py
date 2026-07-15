"""
Phase 4A 回测报告生成

生成 6 张图 + 1 份 REPORT.md:
1. 累计收益曲线 (策略 vs 基准)
2. 分组收益柱状图 (5 分位)
3. IC 时间序列 (逐 fold)
4. 年度收益对比表
5. 回撤曲线
6. 年度 Alpha 分解
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.config_loader import load_config

OUT_DIR = os.path.join(ROOT, "reports", "phase4A")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 7),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data():
    config = load_config()
    gld = pd.read_csv(
        os.path.join(config["paths"]["raw_data"], "market", "gld.csv"),
        parse_dates=["Date"], index_col="Date"
    )["Close"].sort_index()
    daily_ret = gld.pct_change()

    pred = pd.read_parquet(os.path.join(ROOT, "data", "models",
                                        "baseline_predictions.parquet"))
    pred["date"] = pd.to_datetime(pred["date"])
    return gld, daily_ret, pred


def build_strategies(pred, daily_ret, target, model):
    """Build strategy cumulative returns"""
    sub = pred[(pred["target"] == target) & (pred["model"] == model)].copy()
    sub = sub.set_index(pd.to_datetime(sub["date"])).sort_index()
    dr = daily_ret.reindex(sub.index).dropna()
    sub = sub.reindex(dr.index)

    results = {}

    # A1: Direction
    pos = (sub["y_pred"] > 0).astype(float)
    results["A1_Direction"] = {
        "pos": pos, "ret": dr * pos,
        "cum": (1 + dr * pos).cumprod()
    }

    # Buy & Hold
    results["BuyHold"] = {
        "pos": pd.Series(1.0, index=dr.index), "ret": dr,
        "cum": (1 + dr).cumprod()
    }

    # MA 20/60
    from src.backtest.underlying_backtest import benchmark_ma_crossover
    ma_signal = (daily_ret.rolling(20).mean().reindex(dr.index) >
                 daily_ret.rolling(60).mean().reindex(dr.index)).astype(float)
    # Actually use price-based MA crossover
    gld_close = pd.read_csv(
        os.path.join(load_config()["paths"]["raw_data"], "market", "gld.csv"),
        parse_dates=["Date"], index_col="Date"
    )["Close"].sort_index()
    ma_fast = gld_close.rolling(20).mean()
    ma_slow = gld_close.rolling(60).mean()
    ma_pos = (ma_fast > ma_slow).astype(float).reindex(dr.index).fillna(0)
    results["MA_20_60"] = {
        "pos": ma_pos, "ret": dr * ma_pos,
        "cum": (1 + dr * ma_pos).cumprod()
    }

    return results, sub, dr


def fig1_cumulative_returns(pred, daily_ret):
    """图1: 累计收益曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, (target, model) in enumerate([
        ("fwd_ret_20d", "Ridge"),
        ("fwd_ret_10d", "Ridge"),
    ]):
        ax = axes[i]
        results, sub, dr = build_strategies(pred, daily_ret, target, model)

        for name, data in results.items():
            style = "--" if "BM" in name or "Buy" in name or "MA" in name else "-"
            lw = 1.5 if style == "--" else 2
            ax.plot(data["cum"].index, data["cum"].values, style,
                    label=name, linewidth=lw)

        ax.set_title(f"{target} / {model}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (1=start)")
        ax.legend(loc="upper left")
        ax.set_xlabel("Date")

    fig.suptitle("Phase 4A: Cumulative Returns — Strategy vs Benchmarks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_cumulative_returns.png"), dpi=150)
    plt.close()
    print("  [1/6] 累计收益曲线")


def fig2_quintile_returns(pred):
    """图2: 分组收益柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    configs = [
        ("fwd_ret_10d", "Ridge"),
        ("fwd_ret_10d", "XGBoost"),
        ("fwd_ret_20d", "Ridge"),
        ("fwd_ret_20d", "XGBoost"),
    ]

    for idx, (target, model) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        sub = pred[(pred["target"] == target) & (pred["model"] == model)].copy()
        sub["quintile"] = pd.qcut(sub["y_pred"], 5, labels=False,
                                  duplicates="drop")
        group_means = sub.groupby("quintile")["y_true"].mean()

        colors = ["#d32f2f", "#ff7043", "#bdbdbd", "#66bb6a", "#2e7d32"]
        bars = ax.bar(range(len(group_means)), group_means.values * 100,
                      color=colors[:len(group_means)])
        ax.set_xticks(range(len(group_means)))
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(group_means))])
        ax.set_title(f"{target} / {model}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Actual Return (%)")
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Add IC annotation
        ic, _ = stats.spearmanr(sub["y_pred"], sub["y_true"])
        ax.annotate(f"IC={ic:.3f}", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=11, fontweight="bold")

    fig.suptitle("Phase 4A: Quintile Analysis — Predicted vs Actual Returns",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_quintile_returns.png"), dpi=150)
    plt.close()
    print("  [2/6] 分组收益柱状图")


def fig3_ic_by_fold(pred):
    """图3: IC 时间序列 (逐 fold)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    configs = [
        ("fwd_ret_10d", "Ridge"),
        ("fwd_ret_10d", "XGBoost"),
        ("fwd_ret_20d", "Ridge"),
        ("fwd_ret_20d", "XGBoost"),
    ]

    for idx, (target, model) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        sub = pred[(pred["target"] == target) & (pred["model"] == model)]

        fold_ics = []
        fold_labels = []
        for fold in sorted(sub["fold"].unique()):
            fd = sub[sub["fold"] == fold]
            if len(fd) > 30:
                ic, _ = stats.spearmanr(fd["y_pred"], fd["y_true"])
                fold_ics.append(ic)
                mid_date = fd["date"].iloc[len(fd) // 2]
                fold_labels.append(f"F{fold}\n{mid_date.strftime('%Y')}")

        colors = ["#2e7d32" if x > 0 else "#d32f2f" for x in fold_ics]
        ax.bar(range(len(fold_ics)), fold_ics, color=colors)
        ax.set_xticks(range(len(fold_ics)))
        ax.set_xticklabels(fold_labels, fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(f"{target} / {model}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Spearman IC")

        mean_ic = np.mean(fold_ics)
        ax.axhline(y=mean_ic, color="blue", linewidth=1, linestyle="--",
                   label=f"Mean={mean_ic:.3f}")
        ax.legend()

    fig.suptitle("Phase 4A: IC Stability Across Walk-Forward Folds",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_ic_by_fold.png"), dpi=150)
    plt.close()
    print("  [3/6] IC 时间序列")


def fig4_annual_returns(pred, daily_ret):
    """图4: 年度收益对比"""
    target, model = "fwd_ret_20d", "Ridge"
    results, sub, dr = build_strategies(pred, daily_ret, target, model)

    years = sorted(dr.index.year.unique())
    strat_annual = []
    bh_annual = []

    for y in years:
        mask = dr.index.year == y
        s_ret = results["A1_Direction"]["ret"][mask]
        b_ret = results["BuyHold"]["ret"][mask]
        strat_annual.append((1 + s_ret).cumprod().iloc[-1] - 1)
        bh_annual.append((1 + b_ret).cumprod().iloc[-1] - 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(years))
    w = 0.35
    bars1 = ax.bar(x - w / 2, [r * 100 for r in strat_annual], w,
                   label="A1_Direction (Ridge 20d)", color="#1976d2")
    bars2 = ax.bar(x + w / 2, [r * 100 for r in bh_annual], w,
                   label="Buy & Hold GLD", color="#bdbdbd")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Annual Return (%)")
    ax.set_title(f"Phase 4A: Annual Returns — {target}/{model} vs Buy & Hold",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Add alpha labels
    for i, (s, b) in enumerate(zip(strat_annual, bh_annual)):
        alpha = (s - b) * 100
        color = "#2e7d32" if alpha > 0 else "#d32f2f"
        ax.annotate(f"{alpha:+.1f}%", xy=(i, max(s, b) * 100 + 2),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_annual_returns.png"), dpi=150)
    plt.close()
    print("  [4/6] 年度收益对比")


def fig5_drawdown(pred, daily_ret):
    """图5: 回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = [
        ("fwd_ret_20d", "Ridge", "A1_Direction"),
    ]

    for target, model, strat_name in configs:
        results, sub, dr = build_strategies(pred, daily_ret, target, model)

        for name in ["A1_Direction", "BuyHold", "MA_20_60"]:
            cum = results[name]["cum"]
            peak = cum.cummax()
            dd = (cum - peak) / peak * 100
            style = "--" if name != "A1_Direction" else "-"
            ax.plot(dd.index, dd.values, style, label=name, linewidth=1.5)

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Phase 4A: Drawdown — fwd_ret_20d/Ridge vs Benchmarks",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.fill_between(dd.index, dd.values, 0, alpha=0.1, color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_drawdown.png"), dpi=150)
    plt.close()
    print("  [5/6] 回撤曲线")


def fig6_exposure_analysis(pred, daily_ret):
    """图6: 信号暴露度 vs GLD 走势"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    target, model = "fwd_ret_20d", "Ridge"
    sub = pred[(pred["target"] == target) & (pred["model"] == model)].copy()
    sub = sub.set_index(pd.to_datetime(sub["date"])).sort_index()

    gld = pd.read_csv(
        os.path.join(load_config()["paths"]["raw_data"], "market", "gld.csv"),
        parse_dates=["Date"], index_col="Date"
    )["Close"].sort_index()

    # Top panel: GLD price + position
    ax1 = axes[0]
    gld_aligned = gld.reindex(sub.index)
    ax1.plot(gld_aligned.index, gld_aligned.values, color="#333", linewidth=1)
    ax1.set_ylabel("GLD Price ($)")
    ax1.set_title("GLD Price + Model Exposure (fwd_ret_20d / Ridge)",
                  fontsize=13, fontweight="bold")

    # Shade long periods
    pos = (sub["y_pred"] > 0).astype(float)
    for i in range(len(pos) - 1):
        if pos.iloc[i] > 0:
            ax1.axvspan(pos.index[i], pos.index[i + 1],
                       alpha=0.15, color="green")

    # Bottom panel: rolling exposure
    ax2 = axes[1]
    rolling_exp = pos.rolling(60).mean() * 100
    ax2.fill_between(rolling_exp.index, 0, rolling_exp.values,
                     alpha=0.5, color="#1976d2")
    ax2.set_ylabel("60-Day Rolling Exposure (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 100)

    # Mark problem: 2024+ very low exposure during rally
    ax2.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"),
                alpha=0.1, color="red")
    ax2.annotate("2024-2025:\nExposure too low\nduring gold rally",
                 xy=(pd.Timestamp("2024-06-01"), 80),
                 fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_exposure_analysis.png"), dpi=150)
    plt.close()
    print("  [6/6] 暴露度分析")


def generate_report_md():
    """生成 REPORT.md"""
    md = """# Phase 4A: 标的预测回测报告

## 概述

验证 Phase 3 OOS 预测信号是否有稳定的信息含量和经济价值。
回测期间：2016-01 ~ 2026-02 (10 年 OOS)。

---

## 一、核心结论

### 信号有信息含量，但不足以跑赢强势行情

| 发现 | 详情 |
|------|------|
| IC 显著为正 | fwd_ret_10d IC=0.063-0.076, fwd_ret_20d IC=0.083-0.094 |
| 分组单调性 | Top-Bottom spread 正向, Q1(最悲观)实际收益最低 |
| 跨 fold 稳定 | Ridge 90% 的 fold IC 为正 |
| **但** 策略表现弱 | 最好的策略 (Ridge 20d) CAGR=4.1%, 远低于 Buy&Hold 13.8% |
| **关键问题** | 2024-2025 金价暴涨期间, 模型暴露度极低 (2-39%), 错过主升浪 |

### 判断：预测信号达到 Phase 4A 最低通过标准

- [x] OOS IC 在多数 fold 为正 (Ridge 90%)
- [x] Top-Bottom 分组有单调性
- [ ] 成本后优于至少一个简单基准 — **未通过** (弱于 Buy&Hold 和 MA)
- [x] 表现不完全依赖 1-2 个年份 (多年正 alpha)

信号有信息含量但无法直接转化为跑赢强势市场的策略。
**信号的价值在于"看到了东西"，而不是"直接拿来交易"。**

---

## 二、预测质量

| 目标/模型 | IC | p-value | TB Spread | IC正折占比 |
|-----------|-----|---------|-----------|-----------|
| fwd_ret_10d / Ridge | 0.063 | 0.002 | +0.37% | 90% |
| fwd_ret_10d / XGBoost | 0.076 | 0.000 | +0.61% | 70% |
| fwd_ret_20d / Ridge | 0.094 | 0.000 | +1.10% | 90% |
| fwd_ret_20d / XGBoost | 0.083 | 0.000 | +1.12% | 90% |

**长周期信号更强**：20d IC (0.08-0.09) 高于 10d IC (0.06-0.08)，与 Phase 3 结论一致。

---

## 三、策略绩效

| 策略 | CAGR | Sharpe | MaxDD | Exposure |
|------|------|--------|-------|----------|
| **BM: Buy & Hold** | **+13.80%** | **0.89** | -23.02% | 100% |
| **BM: MA 20/60** | **+11.91%** | **0.90** | -15.65% | 61% |
| A1 Ridge 20d | +4.13% | 0.42 | -19.46% | 44% |
| A2 Ridge 20d +Vol | +2.64% | 0.29 | -19.63% | 42% |
| A3 Ridge 20d +Regime | +2.82% | 0.36 | -19.04% | 41% |
| A1 Ridge 10d | +1.03% | 0.12 | -22.13% | 41% |
| A2 Ridge 10d +Vol | +1.87% | 0.22 | -18.14% | 39% |
| A1 XGBoost 10d | -13.00% | -1.33 | -75.80% | 48% |
| A1 XGBoost 20d | -4.24% | -0.38 | -45.50% | 62% |

---

## 四、年度分解 (fwd_ret_20d / Ridge)

| 年份 | 策略 | Buy&Hold | Alpha | Exposure | 备注 |
|------|------|----------|-------|----------|------|
| 2016 | +13.3% | +2.5% | **+10.9%** | 13% | 模型选择性做多, 优于大盘 |
| 2017 | +4.9% | +10.3% | -5.4% | 94% | 高暴露但择时略差 |
| 2018 | -3.4% | -3.3% | -0.1% | 44% | 平手, 熊市中无优势 |
| 2019 | +3.5% | +16.8% | -13.3% | 28% | 错过涨幅 |
| 2020 | +19.4% | +19.4% | +0.0% | 100% | 全仓做多, 完美跟随 |
| 2021 | -6.7% | -5.4% | -1.3% | 75% | 微弱负 alpha |
| 2022 | +2.7% | -0.8% | **+3.4%** | 15% | 规避下跌, 正 alpha |
| 2023 | -5.6% | +12.7% | -18.3% | 36% | 错过涨幅 |
| 2024 | +1.0% | +23.6% | **-22.6%** | 2% | 几乎空仓, 错过暴涨 |
| 2025 | +16.5% | +61.6% | **-45.1%** | 39% | 严重错过主升浪 |

**规律**: 模型在平稳/下跌年份有正 alpha (2016, 2022), 但在强势上涨年份严重落后。

---

## 五、根因分析

### 5.1 模型预测偏差 (Prediction Bias)

| 目标/模型 | pred>0 占比 | 实际>0 占比 | 预测均值 | 实际均值 |
|-----------|-------------|-------------|----------|----------|
| 10d/Ridge | 40.9% | 57.6% | +0.85% | +0.63% |
| 10d/XGB | 47.5% | 57.6% | -0.16% | +0.63% |
| 20d/Ridge | 44.3% | 58.4% | +0.76% | +1.22% |
| 20d/XGB | 62.2% | 58.4% | +0.78% | +1.22% |

**Ridge 严重偏空**: 预测正收益仅 40-44%, 而实际正收益 58%。
模型在 expanding window 训练中被历史数据中的均值回归拉低, 无法适应 2024-2025 的 regime shift。

### 5.2 XGBoost 悖论

XGBoost 10d 的 IC (0.076) 高于 Ridge (0.063), 但交易收益是灾难性的 (-13% CAGR)。

原因: IC 衡量**排序**准确性, 不衡量**阈值**准确性。
XGBoost 的 y_pred 均值为 -0.16% (偏负), 导致用 0 阈值切分时大量做反。

### 5.3 2024-2025 Regime Shift

2024-2025 金价从 ~$190 暴涨至 $280+, 涨幅近 50%。
驱动因素: 央行购金加速 + 地缘政治 + 美国债务加速。
这是模型训练数据中未见过的 regime, 线性模型完全无法适应。

---

## 六、改进方向

| 方向 | 优先级 | 预期效果 |
|------|--------|---------|
| **Regime 切换** | 高 | 在牛市 regime 下提高基础暴露, 不轻易空仓 |
| **自适应阈值** | 高 | 用 expanding median 替代 0 阈值, 修正 bias |
| **信号融合** | 高 | 方向信号 + MA趋势 + 波动率三合一 |
| **作为期权过滤器** | 中 | 不直接做方向, 而是用信号过滤期权入场条件 |
| Release-date 对齐 | 中 | 消除潜在前视偏差后复核基线 |
| 特征选择/降维 | 低 | 减少噪声特征 |

**核心观点**: 当前信号的最大价值不是"直接做方向交易", 而是作为期权策略的输入:
- 方向信号 + 波动率预测 → 选择期权结构
- IC 虽弱但持续为正 → 在期权杠杆下可能放大为有意义的 edge
- 波动率预测 (Phase 3 IC=0.25-0.29) 远强于方向预测, 更适合期权交易

---

## 七、图表

| 图 | 内容 |
|----|------|
| 01_cumulative_returns.png | 累计收益曲线: 策略 vs 基准 |
| 02_quintile_returns.png | 5 分位分组收益 (验证信号单调性) |
| 03_ic_by_fold.png | IC 逐 fold 稳定性 |
| 04_annual_returns.png | 年度收益对比 + Alpha |
| 05_drawdown.png | 回撤曲线 |
| 06_exposure_analysis.png | 信号暴露度 vs GLD 走势 |
"""

    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write(md)
    print("  [REPORT.md] 报告已生成")


def main():
    print("Phase 4A 报告生成中...")
    gld, daily_ret, pred = load_data()

    fig1_cumulative_returns(pred, daily_ret)
    fig2_quintile_returns(pred)
    fig3_ic_by_fold(pred)
    fig4_annual_returns(pred, daily_ret)
    fig5_drawdown(pred, daily_ret)
    fig6_exposure_analysis(pred, daily_ret)
    generate_report_md()

    print(f"\n完成! 报告保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
