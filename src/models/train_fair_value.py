"""
公允价值区间模型 — 训练与评估 (Phase 4A Step 1)

核心思路:
- 预测未来 20 天收益率的分位数 (q10/q25/q50/q75/q90)
- 结合当前 GLD 价格，转换为价格区间
- 评估: 覆盖率、校准、区间宽度、触边回归行为

用法:
    conda activate gold
    python src/models/train_fair_value.py
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
from src.models.data_utils import load_dataset, prepare_Xy, WalkForwardSplitter
from src.models.fair_value_band import (
    CICCQuantile, RidgeQuantile, XGBQuantile, QUANTILES,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gld_prices(config: dict) -> pd.Series:
    """加载 GLD 收盘价"""
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    return gld["Close"].rename("gld_close")


def evaluate_return_quantiles(y_true: np.ndarray,
                              band_df: pd.DataFrame) -> dict:
    """
    评估收益率分位数预测质量。

    y_true: 实际未来收益率
    band_df: 预测的分位数 (q10/q25/q50/q75/q90 为收益率)
    """
    n = len(y_true)
    if n < 30:
        return {}

    results = {}

    # 1. 各分位数校准: 实际低于 qX 的比例应接近 X%
    for q in QUANTILES:
        col = f"q{int(q*100)}"
        if col in band_df.columns:
            actual_below = (y_true < band_df[col].values).mean()
            results[f"coverage_q{int(q*100)}"] = actual_below
            results[f"calib_err_q{int(q*100)}"] = abs(actual_below - q)

    # 2. 80% 区间覆盖率 (q10 ~ q90): 应接近 80%
    in_80 = ((y_true >= band_df["q10"].values) &
             (y_true <= band_df["q90"].values))
    results["coverage_80"] = in_80.mean()

    # 3. 50% 区间覆盖率 (q25 ~ q75): 应接近 50%
    in_50 = ((y_true >= band_df["q25"].values) &
             (y_true <= band_df["q75"].values))
    results["coverage_50"] = in_50.mean()

    # 4. 区间宽度 (收益率空间)
    results["bw_80"] = (band_df["q90"] - band_df["q10"]).mean()
    results["bw_50"] = (band_df["q75"] - band_df["q25"]).mean()

    # 5. q50 (中位数预测) 的质量
    q50 = band_df["q50"].values
    results["q50_mae"] = np.mean(np.abs(y_true - q50))
    ic, _ = stats.spearmanr(q50, y_true)
    results["q50_ic"] = ic

    # 6. Pinball loss (proper scoring for quantile prediction)
    total_pinball = 0
    for q in QUANTILES:
        col = f"q{int(q*100)}"
        residual = y_true - band_df[col].values
        pinball = np.where(residual >= 0, q * residual,
                           (q - 1) * residual).mean()
        results[f"pinball_q{int(q*100)}"] = pinball
        total_pinball += pinball
    results["pinball_avg"] = total_pinball / len(QUANTILES)

    results["N"] = n
    return results


def evaluate_price_band_reversion(gld_close: pd.Series,
                                  price_band: pd.DataFrame,
                                  horizons: list = [5, 10, 20]) -> dict:
    """
    评估价格区间的触边回归行为。

    当 GLD 价格接近 q10/q90 价格边界时，未来 N 天的回归行为。
    """
    results = {}

    # 价格在区间中的位置 (0=q10, 1=q90)
    bw = price_band["q90"] - price_band["q10"]
    position = (gld_close - price_band["q10"]) / bw.replace(0, np.nan)

    near_bottom = position < 0.1
    near_top = position > 0.9
    below_band = position < 0.0
    above_band = position > 1.0
    in_middle = (position >= 0.25) & (position <= 0.75)

    for h in horizons:
        fwd_ret = gld_close.pct_change(h).shift(-h)

        for label, mask in [("bottom", near_bottom), ("top", near_top),
                            ("below", below_band), ("above", above_band),
                            ("middle", in_middle)]:
            valid = mask & fwd_ret.notna()
            if valid.sum() > 5:
                results[f"{label}_ret_{h}d"] = fwd_ret[valid].mean()
                results[f"{label}_count_{h}d"] = int(valid.sum())

    return results


def run_fair_value_evaluation():
    """运行公允价值区间模型评估"""
    config = load_config()
    features, labels = load_dataset(config)
    gld_close = load_gld_prices(config)

    print("=" * 70)
    print("  Phase 4A Step 1: 公允价值区间模型评估")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 预测目标: 未来 20 天收益率
    target = "fwd_ret_20d"
    X, y = prepare_Xy(features, labels, target)

    # 对齐 GLD 价格
    common_idx = X.index.intersection(gld_close.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    gld_close = gld_close.loc[common_idx]

    print(f"  目标: {target}")
    print(f"  样本: {len(X)}, 特征: {X.shape[1]}")
    print(f"  日期: {X.index.min().date()} ~ {X.index.max().date()}")
    print(f"  GLD: ${gld_close.min():.2f} ~ ${gld_close.max():.2f}")

    # Walk-forward
    splitter = WalkForwardSplitter(
        min_train_days=1260, test_days=252, step_days=252)
    folds = splitter.split(X, y)

    # 模型
    models_config = [
        ("CICC-Quantile", lambda: CICCQuantile(alpha=0.1)),
        ("Ridge-Quantile", lambda: RidgeQuantile(alpha=0.01)),
        ("XGB-Quantile", lambda: XGBQuantile()),
    ]

    all_results = []
    all_predictions = []

    for model_name, model_factory in models_config:
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")

        fold_metrics = []
        fold_preds = []

        for fold in folds:
            try:
                m = model_factory()
                m.fit(fold.X_train, fold.y_train)

                # 预测收益率分位数
                ret_band = m.predict(fold.X_test)

                # 评估收益率分位数质量
                metrics = evaluate_return_quantiles(
                    fold.y_test.values, ret_band)
                metrics["fold"] = fold.fold_id
                metrics["model"] = model_name
                metrics["test_start"] = fold.test_start
                metrics["test_end"] = fold.test_end

                # 转换为价格区间并评估触边回归
                close_test = gld_close.loc[fold.X_test.index]
                price_band = m.predict_price_band(fold.X_test, close_test)
                edge = evaluate_price_band_reversion(close_test, price_band)
                metrics.update(edge)

                fold_metrics.append(metrics)

                # 保存预测
                pred_df = pd.DataFrame({
                    "date": fold.X_test.index,
                    "model": model_name,
                    "fold": fold.fold_id,
                    "gld_close": close_test.values,
                    "y_true": fold.y_test.values,
                    "ret_q10": ret_band["q10"].values,
                    "ret_q25": ret_band["q25"].values,
                    "ret_q50": ret_band["q50"].values,
                    "ret_q75": ret_band["q75"].values,
                    "ret_q90": ret_band["q90"].values,
                    "price_q10": price_band["q10"].values,
                    "price_q25": price_band["q25"].values,
                    "price_q50": price_band["q50"].values,
                    "price_q75": price_band["q75"].values,
                    "price_q90": price_band["q90"].values,
                })
                fold_preds.append(pred_df)

            except Exception as e:
                logger.warning(f"  {model_name} fold {fold.fold_id}: {e}")
                import traceback
                traceback.print_exc()
                fold_metrics.append({
                    "fold": fold.fold_id, "model": model_name,
                    "coverage_80": np.nan, "N": 0,
                })

        all_results.extend(fold_metrics)
        if fold_preds:
            all_predictions.append(pd.concat(fold_preds, ignore_index=True))

        # 打印模型摘要
        valid = [m for m in fold_metrics
                 if not np.isnan(m.get("coverage_80", np.nan))]
        if valid:
            _print_model_summary(model_name, valid)

    # 保存
    results_df = pd.DataFrame(all_results)
    preds_df = (pd.concat(all_predictions, ignore_index=True)
                if all_predictions else pd.DataFrame())

    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_parquet(os.path.join(out_dir, "fair_value_results.parquet"))
    if not preds_df.empty:
        preds_df.to_parquet(
            os.path.join(out_dir, "fair_value_predictions.parquet"))

    # 总结表
    _print_comparison_table(results_df)

    print(f"\n  结果已保存到 {out_dir}/fair_value_*.parquet")
    print(f"  共 {len(results_df)} 条评估记录, {len(preds_df)} 条预测")


def _print_model_summary(name: str, valid: list):
    """打印单个模型的摘要"""
    cov80 = np.mean([m["coverage_80"] for m in valid])
    cov50 = np.mean([m["coverage_50"] for m in valid])
    bw80 = np.mean([m["bw_80"] for m in valid])
    bw50 = np.mean([m["bw_50"] for m in valid])
    ic = np.mean([m["q50_ic"] for m in valid])
    pinball = np.mean([m["pinball_avg"] for m in valid])

    calib_errs = []
    for q in [10, 25, 50, 75, 90]:
        errs = [m.get(f"calib_err_q{q}", 0) for m in valid]
        calib_errs.append(np.mean(errs))
    avg_calib = np.mean(calib_errs)

    print(f"\n  区间质量:")
    print(f"    80%覆盖率:  {cov80:.1%}  (目标 80%)")
    print(f"    50%覆盖率:  {cov50:.1%}  (目标 50%)")
    print(f"    80%区间宽度: {bw80:.4f}  (收益率)")
    print(f"    50%区间宽度: {bw50:.4f}  (收益率)")
    print(f"    q50 IC:     {ic:+.4f}")
    print(f"    Pinball loss: {pinball:.6f}")
    print(f"    平均校准误差: {avg_calib:.3f}")

    # 触边回归
    for h in [5, 10, 20]:
        bots = [m.get(f"bottom_ret_{h}d", np.nan) for m in valid]
        tops = [m.get(f"top_ret_{h}d", np.nan) for m in valid]
        mids = [m.get(f"middle_ret_{h}d", np.nan) for m in valid]
        bot = np.nanmean(bots)
        top = np.nanmean(tops)
        mid = np.nanmean(mids)
        if not (np.isnan(bot) and np.isnan(top)):
            print(f"    触边{h:2d}d回归: 底部{bot:+.2%}  "
                  f"顶部{top:+.2%}  中间{mid:+.2%}")


def _print_comparison_table(results_df: pd.DataFrame):
    """打印模型对比总结表"""
    print("\n" + "=" * 70)
    print("  模型对比总结 (目标: fwd_ret_20d 分位数)")
    print("=" * 70)

    # 区间覆盖与校准
    print(f"\n  {'模型':18s} {'80%覆盖':>8s} {'50%覆盖':>8s} "
          f"{'80%宽度':>9s} {'IC':>8s} {'Pinball':>9s} {'校准误':>7s}")
    print(f"  {'-'*67}")

    for name in results_df["model"].unique():
        m_df = results_df[results_df["model"] == name].dropna(
            subset=["coverage_80"])
        if m_df.empty:
            continue
        calib_cols = [c for c in m_df.columns if "calib_err" in c]
        print(f"  {name:18s} "
              f"{m_df['coverage_80'].mean():8.1%} "
              f"{m_df['coverage_50'].mean():8.1%} "
              f"{m_df['bw_80'].mean():9.4f} "
              f"{m_df['q50_ic'].mean():+8.4f} "
              f"{m_df['pinball_avg'].mean():9.6f} "
              f"{m_df[calib_cols].mean(axis=1).mean():7.3f}")

    # 分位数校准详情
    print(f"\n  分位数校准 (实际 < qX 的比例, 应接近名义值)")
    print(f"  {'模型':18s} {'q10':>7s} {'q25':>7s} {'q50':>7s} "
          f"{'q75':>7s} {'q90':>7s}")
    print(f"  {'-'*53}")

    for name in results_df["model"].unique():
        m_df = results_df[results_df["model"] == name].dropna(
            subset=["coverage_80"])
        if m_df.empty:
            continue
        vals = []
        for q in [10, 25, 50, 75, 90]:
            col = f"coverage_q{q}"
            vals.append(f"{m_df[col].mean():6.1%}" if col in m_df else "   N/A")
        print(f"  {name:18s} {vals[0]} {vals[1]} {vals[2]} "
              f"{vals[3]} {vals[4]}")
    print(f"  {'名义值':18s} {'10.0%':>7s} {'25.0%':>7s} {'50.0%':>7s} "
          f"{'75.0%':>7s} {'90.0%':>7s}")

    # 触边回归
    print(f"\n  触边回归 (当前价格触及区间边缘后的未来收益)")
    print(f"  {'模型':18s} {'底5d':>7s} {'底10d':>7s} {'底20d':>7s} "
          f"{'顶5d':>7s} {'顶10d':>7s} {'顶20d':>7s}")
    print(f"  {'-'*62}")

    for name in results_df["model"].unique():
        m_df = results_df[results_df["model"] == name]
        vals = []
        for pos in ["bottom", "top"]:
            for h in [5, 10, 20]:
                col = f"{pos}_ret_{h}d"
                if col in m_df.columns:
                    v = m_df[col].mean()
                    vals.append(f"{v:+6.2%}" if not np.isnan(v) else "   N/A")
                else:
                    vals.append("   N/A")
        print(f"  {name:18s} {vals[0]} {vals[1]} {vals[2]} "
              f"{vals[3]} {vals[4]} {vals[5]}")

    # 区间外统计
    print(f"\n  区间外 (价格在 q10-q90 外的未来收益)")
    print(f"  {'模型':18s} {'下方5d':>7s} {'下方10d':>8s} {'下方20d':>8s} "
          f"{'上方5d':>7s} {'上方10d':>8s} {'上方20d':>8s}")
    print(f"  {'-'*65}")

    for name in results_df["model"].unique():
        m_df = results_df[results_df["model"] == name]
        vals = []
        for pos in ["below", "above"]:
            for h in [5, 10, 20]:
                col = f"{pos}_ret_{h}d"
                if col in m_df.columns:
                    v = m_df[col].mean()
                    vals.append(f"{v:+7.2%}" if not np.isnan(v) else "    N/A")
                else:
                    vals.append("    N/A")
        print(f"  {name:18s} {vals[0]} {vals[1]} {vals[2]} "
              f"{vals[3]} {vals[4]} {vals[5]}")


if __name__ == "__main__":
    run_fair_value_evaluation()
