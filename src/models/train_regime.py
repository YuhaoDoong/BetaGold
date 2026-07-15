"""
Regime 分类器评估 (Phase 4A Step 2)

验证:
1. 各 regime 下 GLD 的收益率、波动率是否有统计显著差异
2. Regime 切换时点 vs 金价趋势转折的关系
3. Regime 分类的稳定性 (不能频繁跳动)

用法:
    conda activate gold
    python src/models/train_regime.py
"""

import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gld_prices(config: dict) -> pd.Series:
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    return gld["Close"].rename("gld_close")


def run_regime_evaluation():
    config = load_config()
    features, labels = load_dataset(config)
    gld_close = load_gld_prices(config)

    print("=" * 70)
    print("  Phase 4A Step 2: 宏观 Regime 分类评估")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 对齐
    common = features.index.intersection(gld_close.index)
    features = features.loc[common]
    gld_close = gld_close.loc[common]

    # 计算 GLD 各周期收益率
    for d in [5, 10, 20, 60]:
        gld_close_aligned = gld_close.reindex(features.index)
        features[f"gld_fwd_ret_{d}d"] = (
            gld_close_aligned.pct_change(d).shift(-d))

    # 分类
    clf = RegimeClassifier()
    regime_df = clf.classify(features)

    # 合并
    df = pd.concat([features, regime_df], axis=1)
    df["gld_close"] = gld_close

    print(f"\n  样本: {len(df)}")
    print(f"  日期: {df.index.min().date()} ~ {df.index.max().date()}")

    # ========================================
    # 1. Regime 分布
    # ========================================
    print(f"\n{'='*60}")
    print("  1. Regime 分布")
    print(f"{'='*60}")

    regime_counts = df["regime"].value_counts()
    for r in ["Bull", "Mixed", "Bear"]:
        n = regime_counts.get(r, 0)
        pct = n / len(df) * 100
        print(f"  {r:6s}: {n:5d} 天 ({pct:5.1f}%)")

    # 按年分布
    print(f"\n  按年份分布:")
    print(f"  {'年份':>6s} {'Bull':>7s} {'Mixed':>7s} {'Bear':>7s}")
    print(f"  {'-'*30}")
    yearly = df.groupby(df.index.year)["regime"].value_counts().unstack(
        fill_value=0)
    for year in sorted(yearly.index):
        row = yearly.loc[year]
        total = row.sum()
        vals = []
        for r in ["Bull", "Mixed", "Bear"]:
            n = row.get(r, 0)
            vals.append(f"{n/total:6.0%}")
        print(f"  {year:>6d} {vals[0]} {vals[1]} {vals[2]}")

    # ========================================
    # 2. 各 Regime 下 GLD 收益率
    # ========================================
    print(f"\n{'='*60}")
    print("  2. 各 Regime 下 GLD 收益率")
    print(f"{'='*60}")

    for horizon in [5, 10, 20, 60]:
        col = f"gld_fwd_ret_{horizon}d"
        if col not in df.columns:
            continue

        print(f"\n  未来 {horizon}d 收益率:")
        print(f"  {'Regime':>6s} {'均值':>8s} {'中位数':>8s} {'标准差':>8s} "
              f"{'正比例':>7s} {'Sharpe*':>8s} {'样本':>6s}")
        print(f"  {'-'*55}")

        for r in ["Bull", "Mixed", "Bear"]:
            mask = df["regime"] == r
            rets = df.loc[mask, col].dropna()
            if len(rets) < 30:
                continue
            mean_r = rets.mean()
            med_r = rets.median()
            std_r = rets.std()
            pos_pct = (rets > 0).mean()
            # 年化 Sharpe 近似
            sharpe = mean_r / std_r * np.sqrt(252 / horizon) if std_r > 0 else 0
            print(f"  {r:>6s} {mean_r:+8.3%} {med_r:+8.3%} {std_r:8.3%} "
                  f"{pos_pct:7.1%} {sharpe:+8.2f} {len(rets):>6d}")

        # 统计检验: Bull vs Bear
        bull_rets = df.loc[df["regime"] == "Bull", col].dropna()
        bear_rets = df.loc[df["regime"] == "Bear", col].dropna()
        if len(bull_rets) > 30 and len(bear_rets) > 30:
            t_stat, p_val = stats.ttest_ind(bull_rets, bear_rets)
            print(f"  Bull vs Bear t-test: t={t_stat:.2f}, p={p_val:.4f} "
                  f"{'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")

    # ========================================
    # 3. 各 Regime 下波动率
    # ========================================
    print(f"\n{'='*60}")
    print("  3. 各 Regime 下 GLD 波动率")
    print(f"{'='*60}")

    # 用 20d 实现波动率
    if "rv_20d" in df.columns:
        print(f"\n  20d 实现波动率:")
        print(f"  {'Regime':>6s} {'均值':>8s} {'中位数':>8s} {'样本':>6s}")
        print(f"  {'-'*32}")
        for r in ["Bull", "Mixed", "Bear"]:
            mask = df["regime"] == r
            vol = df.loc[mask, "rv_20d"].dropna()
            if len(vol) > 0:
                print(f"  {r:>6s} {vol.mean():8.1%} {vol.median():8.1%} "
                      f"{len(vol):>6d}")

    # ========================================
    # 4. Regime 切换频率与稳定性
    # ========================================
    print(f"\n{'='*60}")
    print("  4. Regime 切换频率与稳定性")
    print(f"{'='*60}")

    regime_changes = (df["regime"] != df["regime"].shift(1)).sum()
    avg_duration = len(df) / max(regime_changes, 1)
    print(f"  总切换次数: {regime_changes}")
    print(f"  平均持续天数: {avg_duration:.1f}")

    # 各 regime 的平均持续长度
    regime_runs = []
    current_regime = None
    current_count = 0
    for _, row in df.iterrows():
        if row["regime"] != current_regime:
            if current_regime is not None:
                regime_runs.append(
                    {"regime": current_regime, "duration": current_count})
            current_regime = row["regime"]
            current_count = 1
        else:
            current_count += 1
    if current_regime is not None:
        regime_runs.append(
            {"regime": current_regime, "duration": current_count})

    runs_df = pd.DataFrame(regime_runs)
    if not runs_df.empty:
        print(f"\n  各 Regime 平均持续天数:")
        for r in ["Bull", "Mixed", "Bear"]:
            r_runs = runs_df[runs_df["regime"] == r]["duration"]
            if len(r_runs) > 0:
                print(f"  {r:>6s}: 平均 {r_runs.mean():.0f}d, "
                      f"中位数 {r_runs.median():.0f}d, "
                      f"最长 {r_runs.max():.0f}d, 共 {len(r_runs)} 段")

    # ========================================
    # 5. Regime Score 分布
    # ========================================
    print(f"\n{'='*60}")
    print("  5. Regime Score 分布")
    print(f"{'='*60}")

    score = df["regime_score"]
    print(f"  均值: {score.mean():.4f}")
    print(f"  标准差: {score.std():.4f}")
    print(f"  分位数: [{score.quantile(0.05):.3f}, {score.quantile(0.25):.3f}, "
          f"{score.quantile(0.5):.3f}, {score.quantile(0.75):.3f}, "
          f"{score.quantile(0.95):.3f}]")

    # ========================================
    # 6. 各因子子分数的贡献
    # ========================================
    print(f"\n{'='*60}")
    print("  6. 各因子子分数与 GLD 20d 收益率的相关性")
    print(f"{'='*60}")

    factor_cols = ["fed_rate_direction", "real_yield_level", "usd_trend",
                   "central_bank", "risk_sentiment", "inflation"]
    ret_col = "gld_fwd_ret_20d"
    if ret_col in df.columns:
        print(f"\n  {'因子':25s} {'与20d收益相关':>12s} {'p值':>8s} {'权重':>6s}")
        print(f"  {'-'*55}")
        for fc in factor_cols:
            if fc in df.columns:
                valid = df[[fc, ret_col]].dropna()
                if len(valid) > 100:
                    corr, p = stats.spearmanr(valid[fc], valid[ret_col])
                    w = RegimeClassifier.FACTOR_WEIGHTS.get(fc, 0)
                    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                    print(f"  {fc:25s} {corr:+12.4f} {p:8.4f} {w:5.0%} {sig}")

        # 综合分数
        valid = df[["regime_score", ret_col]].dropna()
        if len(valid) > 100:
            corr, p = stats.spearmanr(valid["regime_score"], valid[ret_col])
            print(f"  {'regime_score (综合)':25s} {corr:+12.4f} {p:8.4f}")

    # ========================================
    # 7. Regime 在关键年份的表现
    # ========================================
    print(f"\n{'='*60}")
    print("  7. 关键年份 Regime 判断回顾")
    print(f"{'='*60}")

    key_years = {
        2013: "金价暴跌 (QE taper恐慌)",
        2016: "Brexit + 降息预期",
        2019: "降息周期开始",
        2020: "COVID + 无限QE",
        2022: "激进加息",
        2024: "央行购金潮 + 降息预期",
        2025: "关税战 + 避险需求",
    }

    for year, desc in key_years.items():
        year_df = df[df.index.year == year]
        if year_df.empty:
            continue
        regime_pcts = year_df["regime"].value_counts(normalize=True)
        gld_start = year_df["gld_close"].iloc[0]
        gld_end = year_df["gld_close"].iloc[-1]
        yr_ret = gld_end / gld_start - 1
        dominant = regime_pcts.index[0]
        dom_pct = regime_pcts.iloc[0]
        avg_score = year_df["regime_score"].mean()

        print(f"\n  {year} — {desc}")
        print(f"    GLD: ${gld_start:.0f} → ${gld_end:.0f} ({yr_ret:+.1%})")
        print(f"    Regime: {dominant} ({dom_pct:.0%}), "
              f"score={avg_score:+.3f}")
        for r in ["Bull", "Mixed", "Bear"]:
            if r in regime_pcts:
                print(f"      {r}: {regime_pcts[r]:.0%}", end="")
        print()

    # 保存
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)

    # 保存 regime 序列
    regime_out = df[["regime_score", "regime"] +
                    [c for c in df.columns if c in factor_cols]].copy()
    regime_out["gld_close"] = df["gld_close"]
    regime_out.to_parquet(os.path.join(out_dir, "regime_classification.parquet"))

    print(f"\n  结果已保存到 {out_dir}/regime_classification.parquet")


if __name__ == "__main__":
    run_regime_evaluation()
