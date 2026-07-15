"""
数据加载与时间序列拆分工具

提供 walk-forward 时间序列拆分器，确保无未来信息泄漏。
"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# 回归型标签 vs 分类型标签
REGRESSION_TARGETS = [
    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
    "fwd_rv_10d", "fwd_rv_20d",
    "iv_rv_spread", "max_dd_10d",
]

CLASSIFICATION_TARGETS = [
    "direction_5d", "direction_10d",
    "tail_event_flag",
]


@dataclass
class FoldData:
    """一次 walk-forward fold 的数据"""
    fold_id: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_dataset(config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载特征和标签"""
    config = config or load_config()
    proc = config["paths"]["processed_data"]
    features = pd.read_parquet(os.path.join(proc, "features_all.parquet"))
    labels = pd.read_parquet(os.path.join(proc, "labels.parquet"))
    logger.info(f"加载数据: {features.shape[1]} 特征, "
                f"{labels.shape[1]} 标签, {len(features)} 行")
    return features, labels


def prepare_Xy(features: pd.DataFrame, labels: pd.DataFrame,
               target: str, max_missing_rate: float = 0.3
               ) -> tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和标签，去除高缺失特征和 NaN 行。

    返回对齐后的 (X, y)，无 NaN。
    """
    # 过滤高缺失特征
    miss = features.isnull().mean()
    keep_cols = miss[miss <= max_missing_rate].index.tolist()
    X = features[keep_cols].copy()

    # 分类标签编码
    if target in labels.columns:
        y = labels[target].copy()
    else:
        raise ValueError(f"标签 '{target}' 不存在")

    if hasattr(y, "cat"):
        y = y.cat.codes.replace(-1, np.nan).astype("float64")

    # 取交集日期，去 NaN
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    logger.info(f"  目标={target}, 特征={X.shape[1]}, 样本={len(X)}")
    return X, y


class WalkForwardSplitter:
    """
    Walk-forward (expanding window) 时间序列拆分器。

    参数:
        min_train_days: 最小训练集天数
        test_days: 每个测试窗口天数
        step_days: 每步前进天数 (默认=test_days)
    """

    def __init__(self, min_train_days: int = 1260,
                 test_days: int = 252, step_days: int = None):
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days or test_days

    def split(self, X: pd.DataFrame, y: pd.Series) -> list[FoldData]:
        """生成 walk-forward folds"""
        dates = X.index.sort_values()
        n = len(dates)
        folds = []
        fold_id = 0

        cutoff = self.min_train_days
        while cutoff + self.test_days <= n:
            train_idx = dates[:cutoff]
            test_idx = dates[cutoff:cutoff + self.test_days]

            folds.append(FoldData(
                fold_id=fold_id,
                X_train=X.loc[train_idx],
                y_train=y.loc[train_idx],
                X_test=X.loc[test_idx],
                y_test=y.loc[test_idx],
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
            ))
            fold_id += 1
            cutoff += self.step_days

        # 最后一个不完整 fold (如果有剩余数据)
        if cutoff < n and cutoff > self.min_train_days:
            remaining = dates[cutoff:]
            if len(remaining) >= 60:  # 至少60天才值得做一个fold
                train_idx = dates[:cutoff]
                folds.append(FoldData(
                    fold_id=fold_id,
                    X_train=X.loc[train_idx],
                    y_train=y.loc[train_idx],
                    X_test=X.loc[remaining],
                    y_test=y.loc[remaining],
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=remaining[0],
                    test_end=remaining[-1],
                ))

        logger.info(f"  Walk-forward: {len(folds)} folds, "
                    f"train>={self.min_train_days}d, test={self.test_days}d")
        return folds
