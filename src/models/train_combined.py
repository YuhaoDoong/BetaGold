"""
组合信号评估 (Phase 4A Step 3)

Regime + CICC 公允价值偏离度 → 仓位信号, Walk-forward OOS 回测。

用法:
    conda activate gold
    python src/models/train_combined.py
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
from src.models.combined_signal import FairValueModel, CombinedSignal

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gld_prices(config: dict) -> pd.Series:
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    return gld["Close"].rename("gld_close")


def run_combined_evaluation():
    config = load_config()
    features, _ = load_dataset(config)
    gld_close = load_gld_prices(config)

    print("=" * 70)
    print("  Phase 4A Step 3: 组合信号评估 (Regime + 公允价值偏离度)")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 对齐
    common = features.index.intersection(gld_close.index)
    features = features.loc[common]
    gld_close = gld_close.loc[common]
    log_price = np.log(gld_close)

    # 去掉 NaN 特征行
    valid = features.notna().all(axis=1) & log_price.notna()
    features = features[valid]
    gld_close = gld_close[valid]
    log_price = log_price[valid]

    print(f"  样本: {len(features)}")
    print(f"  日期: {features.index.min().date()} ~ {features.index.max().date()}")

    # Regime (规则模型, 不需要 walk-forward)
    regime_clf = RegimeClassifier()
    regime_df = regime_clf.classify(features)
    regime = regime_df["regime"]

    # Walk-forward fair value model
    dates = features.index.sort_values()
    n = len(dates)
    min_train, test_size, step = 1260, 252, 252

    oos_fair_values = pd.Series(dtype=float, name="fair_value")

    cutoff = min_train
    while cutoff < n:
        train_idx = dates[:cutoff]
        end = min(cutoff + test_size, n)
        test_idx = dates[cutoff:end]

        model = FairValueModel()
        model.fit(features.loc[train_idx], log_price.loc[train_idx])
        fv = model.predict_fair_value(features.loc[test_idx])
        oos_fair_values = pd.concat([oos_fair_values, fv])

        cutoff += step

    # 去重 (最后 fold 可能有重叠)
    oos_fair_values = oos_fair_values[
        ~oos_fair_values.index.duplicated(keep="last")]

    print(f"  OOS 样本: {len(oos_fair_values)}")
    print(f"  OOS 日期: {oos_fair_values.index.min().date()} ~ "
          f"{oos_fair_values.index.max().date()}")

    # 组合信号
    oos_idx = oos_fair_values.index
    signal_gen = CombinedSignal()
    signal = signal_gen.generate(
        regime.loc[oos_idx],
        gld_close.loc[oos_idx],
        oos_fair_values,
    )

    # 添加未来收益
    fwd_1d = gld_close.pct_change().shift(-1)
    fwd_20d = gld_close.pct_change(20).shift(-20)
    signal["fwd_ret_1d"] = fwd_1d.loc[oos_idx]
    signal["fwd_ret_20d"] = fwd_20d.loc[oos_idx]

    # 去掉 dev_pctile 为 NaN 的行 (初始 lookback 期)
    signal_valid = signal.dropna(subset=["dev_pctile"])
    print(f"  有效信号样本: {len(signal_valid)} "
          f"({signal_valid.index.min().date()} ~ {signal_valid.index.max().date()})")

    # ========================================
    # 1. 偏离度分布
    # ========================================
    print(f"\n{'='*60}")
    print("  1. 公允价值偏离度分布")
    print(f"{'='*60}")

    dev = signal_valid["deviation"]
    print(f"  偏离度: mean={dev.mean():+.1%}, std={dev.std():.1%}")
    print(f"  范围: [{dev.min():.1%}, {dev.max():.1%}]")
    print(f"  分位数: 5%={dev.quantile(0.05):.1%}, "
          f"25%={dev.quantile(0.25):.1%}, "
          f"50%={dev.quantile(0.50):.1%}, "
          f"75%={dev.quantile(0.75):.1%}, "
          f"95%={dev.quantile(0.95):.1%}")

    # ========================================
    # 2. 信号分布
    # ========================================
    print(f"\n{'='*60}")
    print("  2. 信号分布")
    print(f"{'='*60}")

    cross = pd.crosstab(signal_valid["regime"], signal_valid["zone"],
                        margins=True)
    for col_order in [["cheap", "neutral", "expensive", "All"]]:
        cols = [c for c in col_order if c in cross.columns]
        cross = cross[cols]
    print(cross.to_string())

    print(f"\n  仓位分布:")
    for pos in sorted(signal_valid["position"].unique()):
        n_pos = (signal_valid["position"] == pos).sum()
        print(f"    仓位 {pos:.1f}: {n_pos:5d} 天 ({n_pos/len(signal_valid):.1%})")

    # ========================================
    # 3. 各区域 20d 收益
    # ========================================
    print(f"\n{'='*60}")
    print("  3. 各 Regime × Zone 的实际 20d 收益")
    print(f"{'='*60}")

    print(f"\n  {'Regime':>8s} {'Zone':>10s} {'20d均值':>8s} {'正比例':>7s} "
          f"{'Sharpe*':>8s} {'样本':>6s} {'仓位':>5s}")
    print(f"  {'-'*58}")

    for regime_val in ["Bull", "Mixed", "Bear"]:
        for zone_val in ["cheap", "neutral", "expensive"]:
            mask = ((signal_valid["regime"] == regime_val) &
                    (signal_valid["zone"] == zone_val))
            if mask.sum() < 20:
                continue
            rets = signal_valid.loc[mask, "fwd_ret_20d"].dropna()
            if len(rets) < 20:
                continue
            pos = signal_valid.loc[mask, "position"].iloc[0]
            sharpe = rets.mean() / rets.std() * np.sqrt(252/20) if rets.std() > 0 else 0
            print(f"  {regime_val:>8s} {zone_val:>10s} {rets.mean():+8.2%} "
                  f"{(rets>0).mean():7.1%} {sharpe:+8.2f} {len(rets):>6d} "
                  f"{pos:>5.1f}")

    # ========================================
    # 4. 策略回测
    # ========================================
    print(f"\n{'='*60}")
    print("  4. 策略回测")
    print(f"{'='*60}")

    daily_ret = signal_valid["fwd_ret_1d"].fillna(0)

    strategies = {
        "A: Regime+偏离度": signal_valid["position"] * daily_ret,
        "B: 纯Regime(Bull=1)": (signal_valid["regime"] == "Bull").astype(float) * daily_ret,
        "C: Buy&Hold": daily_ret,
    }

    print(f"\n  {'策略':25s} {'总收益':>8s} {'年化':>8s} {'Sharpe':>8s} "
          f"{'MaxDD':>8s} {'暴露度':>8s}")
    print(f"  {'-'*68}")

    for name, strat_ret in strategies.items():
        cum = (1 + strat_ret).cumprod()
        total_ret = cum.iloc[-1] - 1
        n_years = (signal_valid.index[-1] - signal_valid.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
        dd = (cum / cum.cummax() - 1).min()
        if "Regime+偏离度" in name:
            exp = (signal_valid["position"] > 0).mean()
        elif "纯Regime" in name:
            exp = (signal_valid["regime"] == "Bull").mean()
        else:
            exp = 1.0
        print(f"  {name:25s} {total_ret:+8.1%} {cagr:+8.1%} {sharpe:+8.2f} "
              f"{dd:+8.1%} {exp:8.1%}")

    # 年度明细
    print(f"\n  年度收益:")
    print(f"  {'年份':>6s} {'Regime+偏离度':>14s} {'纯Regime':>12s} "
          f"{'Buy&Hold':>12s} {'Bull':>6s} {'Cheap':>6s}")
    print(f"  {'-'*58}")

    for year in sorted(signal_valid.index.year.unique()):
        yr_mask = signal_valid.index.year == year
        yr_vals = {}
        for name, strat_ret in strategies.items():
            yr_r = (1 + strat_ret[yr_mask]).cumprod().iloc[-1] - 1
            yr_vals[name] = yr_r
        bull_pct = (signal_valid.loc[yr_mask, "regime"] == "Bull").mean()
        cheap_pct = (signal_valid.loc[yr_mask, "zone"] == "cheap").mean()
        print(f"  {year:>6d} {yr_vals['A: Regime+偏离度']:+14.1%} "
              f"{yr_vals['B: 纯Regime(Bull=1)']:+12.1%} "
              f"{yr_vals['C: Buy&Hold']:+12.1%} "
              f"{bull_pct:6.0%} {cheap_pct:6.0%}")

    # ========================================
    # 5. 核心验证: Bull+cheap vs Bull+expensive
    # ========================================
    print(f"\n{'='*60}")
    print("  5. 核心验证: 偏离度在 Bull regime 下是否有区分力")
    print(f"{'='*60}")

    bull_mask = signal_valid["regime"] == "Bull"
    bull_data = signal_valid[bull_mask]

    if len(bull_data) > 100:
        for label, zone in [("相对便宜(cheap)", "cheap"),
                             ("中性(neutral)", "neutral"),
                             ("相对贵(expensive)", "expensive")]:
            z_mask = bull_data["zone"] == zone
            rets = bull_data.loc[z_mask, "fwd_ret_20d"].dropna()
            if len(rets) < 20:
                print(f"  Bull+{label}: 样本不足 ({len(rets)})")
                continue
            print(f"  Bull+{label}: n={len(rets)}, "
                  f"20d={rets.mean():+.2%}, "
                  f"正比例={(rets>0).mean():.1%}, "
                  f"Sharpe={rets.mean()/rets.std()*np.sqrt(252/20):+.2f}")

        # t-test
        cheap_rets = bull_data.loc[bull_data["zone"] == "cheap", "fwd_ret_20d"].dropna()
        exp_rets = bull_data.loc[bull_data["zone"] == "expensive", "fwd_ret_20d"].dropna()
        if len(cheap_rets) > 20 and len(exp_rets) > 20:
            t, p = stats.ttest_ind(cheap_rets, exp_rets)
            print(f"\n  Bull: cheap vs expensive t-test: t={t:.2f}, p={p:.4f} "
                  f"{'***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''}")

    # 保存
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    signal_valid.to_parquet(os.path.join(out_dir, "combined_signal_oos.parquet"))
    print(f"\n  结果已保存到 {out_dir}/combined_signal_oos.parquet")


if __name__ == "__main__":
    run_combined_evaluation()
