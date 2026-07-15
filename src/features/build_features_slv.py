"""
白银 (SLV) 特征工程

和 build_features.py 同结构, 但以 SLV 为主资产:
- 技术面: 用 slv.csv
- 宏观面: 与 GLD 共享
- 波动率: RV 用 SLV 价格; GVZ/VIX/MOVE 仍沿用 (宏观波动率代理)

输出: data/processed/features_slv.parquet

用法:
    conda activate gold
    python src/features/build_features_slv.py
"""

import os
import sys
import logging
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.features.technical_features import TechnicalFeatureBuilder
from src.features.macro_features import MacroFeatureBuilder
from src.features.vol_features import VolFeatureBuilder


def build_slv_features(config: dict = None) -> pd.DataFrame:
    config = config or load_config()

    print("=" * 60)
    print("  SLV 特征工程")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/3] 技术面特征 (SLV)...")
    tech = TechnicalFeatureBuilder(config, ticker="slv").build()
    print(f"  技术面: {tech.shape[1]} 列")

    print("\n[2/3] 宏观特征 (共享)...")
    macro = MacroFeatureBuilder(config).build()
    print(f"  宏观面: {macro.shape[1]} 列")

    print("\n[3/3] 波动率特征 (SLV RV + 共享 IV)...")
    vol = VolFeatureBuilder(config, ticker="slv").build()
    print(f"  波动率: {vol.shape[1]} 列")

    features = pd.concat([tech, macro, vol], axis=1, join="outer")
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.sort_index()

    print(f"\n  合并结果: {features.shape[1]} 列 x {features.shape[0]} 行")
    return features


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    config = load_config()
    features = build_slv_features(config)

    out_dir = config["paths"]["processed_data"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "features_slv.parquet")
    features.to_parquet(out_path)

    print(f"\n  保存: {out_path}")
    print("SLV 特征工程完成!")


if __name__ == "__main__":
    main()
