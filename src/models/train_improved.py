"""
改进模型统一对比评估

在同一 walk-forward 框架下评估所有模型变体, 输出对比表。

用法:
    conda activate gold
    python src/models/train_improved.py
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
from src.models.data_utils import (
    load_dataset, prepare_Xy, WalkForwardSplitter,
    REGRESSION_TARGETS,
)
from src.models.baseline import RidgeBaseline, XGBBaseline
from src.models.improved_models import (
    CICCModel, RidgeSelected, XGBTuned, GARCHModel,
    RidgeRegime, SimpleEnsemble,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算单 fold 评估指标"""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 30:
        return {"IC": np.nan, "Dir_Acc": np.nan, "MAE": np.nan, "N": valid.sum()}

    yt = y_true[valid]
    yp = y_pred[valid]

    ic, ic_p = stats.spearmanr(yp, yt)
    mae = np.mean(np.abs(yt - yp))

    # 方向准确率
    dir_correct = np.sign(yp) == np.sign(yt)
    dir_acc = dir_correct.mean()

    return {
        "IC": ic,
        "IC_pval": ic_p,
        "Dir_Acc": dir_acc,
        "MAE": mae,
        "N": int(valid.sum()),
    }


def build_models_for_target(target: str) -> list:
    """根据预测目标构建待评估的模型列表"""
    models = []

    # 基线
    models.append(RidgeBaseline(task="regression"))  # Ridge
    models.append(XGBBaseline(task="regression"))     # XGBoost

    # 中金模型
    models.append(CICCModel(n_factors=4))  # CICC 4因子
    models.append(CICCModel(n_factors=3))  # CICC 3因子

    # 改进模型
    models.append(RidgeSelected())         # LASSO特征选择 + Ridge
    models.append(XGBTuned())              # 调参XGBoost
    models.append(RidgeRegime())           # Regime条件Ridge

    # Ensemble
    models.append(SimpleEnsemble([
        RidgeBaseline(task="regression"),
        XGBTuned(),
        CICCModel(n_factors=4),
    ]))

    # GARCH (仅波动率目标)
    if "rv" in target:
        horizon = 10 if "10d" in target else 20
        models.append(GARCHModel(horizon=horizon))

    return models


def run_comparison():
    """运行全部模型对比"""
    config = load_config()
    features, labels = load_dataset(config)

    print("=" * 70)
    print("  Phase 4A: 改进模型对比评估")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    splitter = WalkForwardSplitter(
        min_train_days=1260, test_days=252, step_days=252)

    # 核心回归目标
    targets = [
        "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
        "fwd_rv_10d", "fwd_rv_20d",
    ]

    all_results = []
    all_predictions = []

    for t_idx, target in enumerate(targets):
        print(f"\n{'='*60}")
        print(f"[{t_idx+1}/{len(targets)}] {target}")
        print(f"{'='*60}")

        X, y = prepare_Xy(features, labels, target)
        folds = splitter.split(X, y)
        if not folds:
            continue

        models = build_models_for_target(target)

        for model in models:
            fold_metrics = []
            fold_preds = []

            for fold in folds:
                try:
                    model_copy = _clone_model(model)
                    model_copy.fit(fold.X_train, fold.y_train)
                    pred = model_copy.predict(fold.X_test)

                    metrics = evaluate_fold(fold.y_test.values, pred)
                    metrics["fold"] = fold.fold_id
                    metrics["target"] = target
                    metrics["model"] = model.name
                    metrics["test_start"] = fold.test_start
                    metrics["test_end"] = fold.test_end
                    fold_metrics.append(metrics)

                    # 保存预测
                    pred_df = pd.DataFrame({
                        "date": fold.X_test.index,
                        "target": target,
                        "model": model.name,
                        "y_true": fold.y_test.values,
                        "y_pred": pred,
                        "fold": fold.fold_id,
                    })
                    fold_preds.append(pred_df)

                except Exception as e:
                    logger.warning(f"  {model.name} fold {fold.fold_id}: {e}")
                    fold_metrics.append({
                        "fold": fold.fold_id, "target": target,
                        "model": model.name, "IC": np.nan,
                        "Dir_Acc": np.nan, "MAE": np.nan, "N": 0,
                    })

            all_results.extend(fold_metrics)
            if fold_preds:
                all_predictions.append(pd.concat(fold_preds, ignore_index=True))

            # 打印该模型摘要
            ics = [m["IC"] for m in fold_metrics if not np.isnan(m.get("IC", np.nan))]
            das = [m["Dir_Acc"] for m in fold_metrics if not np.isnan(m.get("Dir_Acc", np.nan))]
            if ics:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics)
                ic_pos = np.mean([x > 0 for x in ics])
                da_mean = np.mean(das) if das else 0
                print(f"  {model.name:20s}  IC={ic_mean:+.4f} ±{ic_std:.4f}  "
                      f"IC+={ic_pos:.0%}  Dir={da_mean:.1%}")

    # 汇总并保存
    results_df = pd.DataFrame(all_results)
    preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)

    results_df.to_parquet(os.path.join(out_dir, "improved_results.parquet"))
    if not preds_df.empty:
        preds_df.to_parquet(os.path.join(out_dir, "improved_predictions.parquet"))

    # 打印总结对比表
    print("\n" + "=" * 70)
    print("  模型对比总结 (平均 IC)")
    print("=" * 70)

    for target in targets:
        t_df = results_df[results_df["target"] == target]
        if t_df.empty:
            continue

        print(f"\n  {target}:")
        print(f"  {'模型':20s} {'IC均值':>8s} {'IC标准差':>8s} {'IC+折':>6s} "
              f"{'Dir_Acc':>8s} {'MAE':>10s}")
        print(f"  {'-'*62}")

        summary = []
        for model_name in t_df["model"].unique():
            m_df = t_df[t_df["model"] == model_name]
            ic_vals = m_df["IC"].dropna()
            da_vals = m_df["Dir_Acc"].dropna()
            mae_vals = m_df["MAE"].dropna()
            if len(ic_vals) == 0:
                continue
            summary.append({
                "model": model_name,
                "ic_mean": ic_vals.mean(),
                "ic_std": ic_vals.std(),
                "ic_pos": (ic_vals > 0).mean(),
                "da_mean": da_vals.mean() if len(da_vals) > 0 else np.nan,
                "mae_mean": mae_vals.mean() if len(mae_vals) > 0 else np.nan,
            })

        # 按 IC 排序
        summary.sort(key=lambda x: x["ic_mean"], reverse=True)
        for s in summary:
            marker = " ***" if s["ic_mean"] == max(x["ic_mean"] for x in summary) else ""
            print(f"  {s['model']:20s} {s['ic_mean']:+8.4f} {s['ic_std']:8.4f} "
                  f"{s['ic_pos']:6.0%} {s['da_mean']:8.1%} "
                  f"{s['mae_mean']:10.6f}{marker}")

    print(f"\n  结果已保存到 {out_dir}/improved_*.parquet")
    print(f"  共 {len(results_df)} 条评估记录, {len(preds_df)} 条预测")


def _clone_model(model):
    """简单克隆模型（重建新实例）"""
    cls = type(model)
    if cls.__name__ == "RidgeBaseline":
        return RidgeBaseline(task="regression")
    elif cls.__name__ == "XGBBaseline":
        return XGBBaseline(task="regression")
    elif cls.__name__ == "CICCModel":
        return CICCModel(n_factors=model.n_factors)
    elif cls.__name__ == "RidgeSelected":
        return RidgeSelected()
    elif cls.__name__ == "XGBTuned":
        return XGBTuned()
    elif cls.__name__ == "GARCHModel":
        return GARCHModel(p=model.p, q=model.q, horizon=model.horizon)
    elif cls.__name__ == "RidgeRegime":
        return RidgeRegime()
    elif cls.__name__ == "SimpleEnsemble":
        # 重建子模型
        sub_models = [_clone_model(m) for m in model.models]
        return SimpleEnsemble(sub_models)
    else:
        raise ValueError(f"Unknown model: {cls.__name__}")


if __name__ == "__main__":
    run_comparison()
