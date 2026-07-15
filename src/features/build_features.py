"""
特征工程主入口 (Phase 2)
合并技术面 + 宏观面 + 波动率特征 + 标签，生成最终建模数据集。

输出:
- data/processed/features_all.parquet  — 全部特征 (不含标签)
- data/processed/labels.parquet        — 全部标签
- data/processed/dataset.parquet       — 特征 + 标签合并

用法:
    conda activate gold
    python src/features/build_features.py
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.features.technical_features import TechnicalFeatureBuilder
from src.features.macro_features import MacroFeatureBuilder
from src.features.vol_features import VolFeatureBuilder
from src.features.label_builder import LabelBuilder

logger = logging.getLogger(__name__)


def build_all_features(config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建全部特征和标签

    返回: (features_df, labels_df)
    """
    config = config or load_config()

    print("=" * 60)
    print("  Phase 2: 特征工程")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 技术面特征
    print("\n[1/4] 构建技术面特征...")
    tech = TechnicalFeatureBuilder(config).build()
    print(f"  技术面: {tech.shape[1]} 列")

    # 2. 宏观面特征
    print("\n[2/4] 构建宏观面特征...")
    macro = MacroFeatureBuilder(config).build()
    print(f"  宏观面: {macro.shape[1]} 列")

    # 3. 波动率特征
    print("\n[3/4] 构建波动率特征...")
    vol = VolFeatureBuilder(config).build()
    print(f"  波动率: {vol.shape[1]} 列")

    # 4. 标签
    print("\n[4/4] 构建预测标签...")
    labels = LabelBuilder(config).build()
    print(f"  标签: {labels.shape[1]} 列")

    # 合并特征 (同 index 对齐)
    features = pd.concat([tech, macro, vol], axis=1, join="outer")

    # 去除完全重复的列名 (不同模块可能重复产生)
    features = features.loc[:, ~features.columns.duplicated()]

    # 排序
    features = features.sort_index()
    labels = labels.sort_index()

    print(f"\n{'=' * 60}")
    print(f"  合并结果")
    print(f"{'=' * 60}")
    print(f"  特征: {features.shape[1]} 列 x {features.shape[0]} 行")
    print(f"  标签: {labels.shape[1]} 列 x {labels.shape[0]} 行")

    # 缺失率统计
    missing = features.isnull().mean()
    high_missing = missing[missing > 0.5]
    if len(high_missing) > 0:
        print(f"\n  高缺失率特征 (>50%): {len(high_missing)} 个")
        for col, rate in high_missing.sort_values(ascending=False).head(5).items():
            print(f"    {col}: {rate:.1%}")

    low_missing = missing[missing <= 0.5]
    avg_missing = low_missing.mean() if len(low_missing) > 0 else 0
    print(f"  平均缺失率 (<=50%的特征): {avg_missing:.2%}")

    return features, labels


def save_features(features: pd.DataFrame, labels: pd.DataFrame,
                  config: dict = None):
    """保存特征和标签到 processed 目录"""
    config = config or load_config()
    output_dir = config["paths"]["processed_data"]
    os.makedirs(output_dir, exist_ok=True)

    # 保存特征
    feat_path = os.path.join(output_dir, "features_all.parquet")
    features.to_parquet(feat_path)
    logger.info(f"已保存特征 -> {feat_path}")

    # 保存标签
    label_path = os.path.join(output_dir, "labels.parquet")
    labels.to_parquet(label_path)
    logger.info(f"已保存标签 -> {label_path}")

    # 合并保存 (方便直接加载)
    # 只合并数值型标签 (分类标签转为数值编码)
    labels_numeric = labels.copy()
    for col in labels_numeric.select_dtypes(include=["category"]).columns:
        labels_numeric[col] = labels_numeric[col].cat.codes
    dataset = pd.concat([features, labels_numeric], axis=1, join="inner")
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    ds_path = os.path.join(output_dir, "dataset.parquet")
    dataset.to_parquet(ds_path)
    logger.info(f"已保存完整数据集 -> {ds_path}")

    # CSV 备份 (只保存特征名单)
    feat_list = pd.DataFrame({
        "feature": features.columns,
        "missing_rate": features.isnull().mean().values,
        "dtype": features.dtypes.astype(str).values,
    })
    feat_list.to_csv(os.path.join(output_dir, "feature_list.csv"), index=False)

    label_list = pd.DataFrame({
        "label": labels.columns,
        "dtype": labels.dtypes.astype(str).values,
    })
    label_list.to_csv(os.path.join(output_dir, "label_list.csv"), index=False)

    print(f"\n  已保存到 {output_dir}/:")
    print(f"    features_all.parquet  ({features.shape[1]} 特征)")
    print(f"    labels.parquet        ({labels.shape[1]} 标签)")
    print(f"    dataset.parquet       ({dataset.shape[1]} 列合并)")
    print(f"    feature_list.csv      (特征名单+缺失率)")
    print(f"    label_list.csv        (标签名单)")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    config = load_config()
    features, labels = build_all_features(config)
    save_features(features, labels, config)

    print("\n特征工程完成!")


if __name__ == "__main__":
    main()
