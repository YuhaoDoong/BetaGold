"""
Phase 3: 基线模型训练主入口

Walk-forward 评估 Ridge + XGBoost 在全部预测目标上的表现。

输出:
- data/models/baseline_results.parquet — 逐 fold 评估结果
- data/models/baseline_predictions.parquet — 逐日 OOS 预测
- data/models/feature_importance.parquet — XGBoost 特征重要性

用法:
    conda activate gold
    python src/models/train_baseline.py
"""

import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import (
    load_dataset, prepare_Xy, WalkForwardSplitter,
    REGRESSION_TARGETS, CLASSIFICATION_TARGETS,
)
from src.models.baseline import RidgeBaseline, XGBBaseline
from src.models.evaluation import (
    regression_metrics, classification_metrics, summarize_fold_results,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def train_one_target(features: pd.DataFrame, labels: pd.DataFrame,
                     target: str, splitter: WalkForwardSplitter,
                     ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    对一个目标标签训练 Ridge + XGBoost，walk-forward 评估。

    返回: (fold_results, oos_predictions, feature_importances)
    """
    is_clf = target in CLASSIFICATION_TARGETS
    task = "classification" if is_clf else "regression"

    X, y = prepare_Xy(features, labels, target)
    folds = splitter.split(X, y)

    if len(folds) == 0:
        logger.warning(f"  {target}: 数据不足，跳过")
        return [], pd.DataFrame(), pd.DataFrame()

    fold_results = []
    all_preds = []
    all_importances = []

    models = [
        RidgeBaseline(task=task),
        XGBBaseline(task=task),
    ]

    for fold in folds:
        for model in models:
            model.fit(fold.X_train, fold.y_train)
            pred = model.predict(fold.X_test)
            prob = model.predict_proba(fold.X_test) if is_clf else None

            # 评估
            if is_clf:
                metrics = classification_metrics(
                    fold.y_test.values, pred, prob)
            else:
                metrics = regression_metrics(fold.y_test.values, pred)

            metrics["target"] = target
            metrics["model"] = model.name
            metrics["fold"] = fold.fold_id
            metrics["train_end"] = fold.train_end
            metrics["test_start"] = fold.test_start
            metrics["test_end"] = fold.test_end
            metrics["train_size"] = len(fold.X_train)
            fold_results.append(metrics)

            # 保存 OOS 预测
            pred_df = pd.DataFrame({
                "date": fold.X_test.index,
                "target": target,
                "model": model.name,
                "y_true": fold.y_test.values,
                "y_pred": pred,
                "fold": fold.fold_id,
            })
            all_preds.append(pred_df)

            # XGBoost 特征重要性
            if isinstance(model, XGBBaseline):
                imp = model.feature_importance(top_n=30)
                imp_df = pd.DataFrame({
                    "feature": imp.index,
                    "importance": imp.values,
                    "target": target,
                    "fold": fold.fold_id,
                })
                all_importances.append(imp_df)

    oos_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    feat_imp = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    return fold_results, oos_preds, feat_imp


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    config = load_config()
    features, labels = load_dataset(config)

    print("=" * 60)
    print("  Phase 3: 基线模型训练")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    splitter = WalkForwardSplitter(
        min_train_days=1260,  # 5 年
        test_days=252,        # 1 年
        step_days=252,        # 步进 1 年
    )

    # 选择核心目标 (避免全跑太慢)
    targets = [
        # 回归
        "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
        "fwd_rv_10d", "fwd_rv_20d",
        # 分类
        "direction_5d", "direction_10d",
        "tail_event_flag",
    ]

    all_results = []
    all_preds = []
    all_importances = []

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target}...")
        results, preds, imp = train_one_target(
            features, labels, target, splitter)
        all_results.extend(results)

        if not preds.empty:
            all_preds.append(preds)
        if not imp.empty:
            all_importances.append(imp)

        # 打印该目标摘要
        if results:
            df = pd.DataFrame(results)
            for model_name in df["model"].unique():
                m = df[df["model"] == model_name]
                key_metric = "Accuracy" if target in CLASSIFICATION_TARGETS else "IC"
                if key_metric in m.columns:
                    vals = m[key_metric].dropna()
                    if len(vals) > 0:
                        print(f"  {model_name}: {key_metric} = "
                              f"{vals.mean():.4f} +/- {vals.std():.4f}")

    # 保存结果
    output_dir = os.path.join(config["paths"].get("models_data",
                              os.path.join(PROJECT_ROOT, "data", "models")))
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(output_dir, "baseline_results.parquet"))

    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_df.to_parquet(os.path.join(output_dir, "baseline_predictions.parquet"))

    if all_importances:
        imp_df = pd.concat(all_importances, ignore_index=True)
        imp_df.to_parquet(os.path.join(output_dir, "baseline_importances.parquet"))

    # 汇总表
    print(f"\n{'=' * 60}")
    print("  基线模型汇总")
    print(f"{'=' * 60}")

    for target in targets:
        t_df = results_df[results_df["target"] == target]
        if t_df.empty:
            continue
        is_clf = target in CLASSIFICATION_TARGETS
        key = "Accuracy" if is_clf else "IC"
        if key not in t_df.columns:
            continue
        print(f"\n  {target} ({key}):")
        for model_name in ["Ridge", "LogReg", "XGBoost"]:
            m = t_df[t_df["model"] == model_name]
            if m.empty or key not in m.columns:
                continue
            vals = m[key].dropna()
            if len(vals) > 0:
                print(f"    {model_name:8s}: {vals.mean():.4f} "
                      f"(std={vals.std():.4f}, "
                      f"min={vals.min():.4f}, max={vals.max():.4f})")

    # 方向准确率 (回归)
    reg_targets = [t for t in targets if t in REGRESSION_TARGETS]
    if reg_targets:
        print(f"\n  方向准确率 (Dir_Acc):")
        for target in reg_targets:
            t_df = results_df[results_df["target"] == target]
            if t_df.empty or "Dir_Acc" not in t_df.columns:
                continue
            for model_name in ["Ridge", "XGBoost"]:
                m = t_df[t_df["model"] == model_name]
                if m.empty:
                    continue
                vals = m["Dir_Acc"].dropna()
                if len(vals) > 0:
                    print(f"    {target} / {model_name:8s}: "
                          f"{vals.mean():.1%} +/- {vals.std():.1%}")

    print(f"\n  结果已保存到 {output_dir}/")
    print(f"    baseline_results.parquet     ({len(results_df)} 条)")
    if all_preds:
        print(f"    baseline_predictions.parquet ({len(preds_df)} 条)")
    if all_importances:
        print(f"    baseline_importances.parquet ({len(imp_df)} 条)")

    print("\nPhase 3 基线训练完成!")


if __name__ == "__main__":
    main()
