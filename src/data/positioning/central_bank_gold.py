"""
央行购金数据模块
解析世界黄金协会(WGC)下载的Excel数据，提取全球央行月度/年度净购金量
数据来源: https://china.gold.org/goldhub/data/gold-reserves-by-country
需要手动下载Excel后放入 data/raw/others/ 目录
"""

import os
import logging

import pandas as pd
import numpy as np

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# WGC Excel 文件的格式定义
CHANGES_MONTHLY_HEADER_ROW = 7  # 第7行(0-indexed)是日期表头
CHANGES_DATA_START_ROW = 8      # 第8行起是国家数据
COUNTRY_COL = 1                 # 第1列是国家名
DATE_START_COL = 3              # 第3列起是月度数据

# 主要购金央行（用于分国别分析）
KEY_CENTRAL_BANKS = [
    "China, P.R.: Mainland",
    "Russian Federation",
    "India",
    "Poland, Rep. of",
    "Singapore",
    "Kazakhstan, Rep. of",
    "Uzbekistan, Rep. of",
    "Czech Republic",
    "Qatar",
    "Hungary",
]


class CentralBankGoldCollector:
    """央行购金数据采集器（基于WGC Excel）"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        self.others_dir = os.path.join(self.raw_data_path, "others")

    def _find_changes_file(self) -> str | None:
        """查找 Changes Excel 文件"""
        if not os.path.exists(self.others_dir):
            return None
        for f in os.listdir(self.others_dir):
            if f.startswith("Changes") and f.endswith(".xlsx"):
                return os.path.join(self.others_dir, f)
        return None

    def _find_holdings_file(self) -> str | None:
        """查找 Holdings Excel 文件"""
        if not os.path.exists(self.others_dir):
            return None
        for f in os.listdir(self.others_dir):
            if f.startswith("World_official") and f.endswith(".xlsx"):
                return os.path.join(self.others_dir, f)
        return None

    def parse_monthly_changes(self) -> pd.DataFrame:
        """
        解析月度央行购金变化数据
        返回 DataFrame: 行=月份, 列=各国购金量(吨) + 全球总量
        """
        filepath = self._find_changes_file()
        if filepath is None:
            logger.error(
                "未找到 WGC Changes Excel 文件。\n"
                "请从 https://china.gold.org/goldhub/data/gold-reserves-by-country 下载\n"
                "并放入 data/raw/others/ 目录"
            )
            return pd.DataFrame()

        logger.info(f"解析月度央行购金数据: {filepath}")

        df = pd.read_excel(filepath, sheet_name="Monthly", header=None)

        # 提取日期（第7行，从第3列开始）
        dates = pd.to_datetime(df.iloc[CHANGES_MONTHLY_HEADER_ROW, DATE_START_COL:].values,
                               errors="coerce")

        # 提取国家名
        countries = df.iloc[CHANGES_DATA_START_ROW:, COUNTRY_COL].values

        # 提取数据块
        data_block = df.iloc[CHANGES_DATA_START_ROW:, DATE_START_COL:].copy()
        data_block.columns = dates
        data_block.index = countries
        data_block = data_block.apply(pd.to_numeric, errors="coerce")

        # 去掉日期为NaT的列
        data_block = data_block.loc[:, data_block.columns.notna()]

        # 转置：行=日期, 列=国家
        result = data_block.T
        result.index.name = "Date"

        # 添加全球总量列
        result["Global_Total"] = result.sum(axis=1)

        # 添加主要购金国小计
        key_cols = [c for c in KEY_CENTRAL_BANKS if c in result.columns]
        if key_cols:
            result["Key_CBs_Total"] = result[key_cols].sum(axis=1)

        logger.info(f"月度数据: {len(result)} 个月, {len(result.columns)} 列, "
                    f"{result.index[0].strftime('%Y-%m')} ~ {result.index[-1].strftime('%Y-%m')}")

        return result

    def parse_annual_changes(self) -> pd.DataFrame:
        """解析年度央行购金变化数据"""
        filepath = self._find_changes_file()
        if filepath is None:
            return pd.DataFrame()

        logger.info(f"解析年度央行购金数据: {filepath}")

        df = pd.read_excel(filepath, sheet_name="Annual", header=None)

        # 第0行是表头(Country, Comments, 年份...)
        years = df.iloc[0, 2:].values
        countries = df.iloc[1:, 0].values

        data_block = df.iloc[1:, 2:].copy()
        data_block.columns = [int(y) for y in years if pd.notna(y)]
        data_block.index = countries
        data_block = data_block.apply(pd.to_numeric, errors="coerce")

        result = data_block.T
        result.index.name = "Year"
        result["Global_Total"] = result.sum(axis=1)

        logger.info(f"年度数据: {len(result)} 年, {result.index[0]} ~ {result.index[-1]}")

        return result

    def parse_current_holdings(self) -> pd.DataFrame:
        """解析当前各国央行黄金持有量"""
        filepath = self._find_holdings_file()
        if filepath is None:
            logger.warning("未找到 WGC Holdings Excel 文件")
            return pd.DataFrame()

        logger.info(f"解析当前黄金持有量: {filepath}")

        df = pd.read_excel(filepath, sheet_name="PDF", header=None)

        # 数据从第6行开始（0-indexed），格式: 排名, 国家, 吨数, 占比, 日期
        records = []
        for i in range(6, len(df)):
            row = df.iloc[i]
            # 左半部分
            if pd.notna(row.iloc[1]) and pd.notna(row.iloc[2]):
                records.append({
                    "Country": str(row.iloc[1]).strip(),
                    "Holdings_Tonnes": float(row.iloc[2]) if pd.notna(row.iloc[2]) else None,
                    "Pct_of_Reserves": float(row.iloc[3]) if pd.notna(row.iloc[3]) else None,
                })
            # 右半部分
            if len(row) > 7 and pd.notna(row.iloc[7]) and pd.notna(row.iloc[7]):
                records.append({
                    "Country": str(row.iloc[7]).strip(),
                    "Holdings_Tonnes": float(row.iloc[7]) if isinstance(row.iloc[7], (int, float)) else None,
                })

        result = pd.DataFrame(records).dropna(subset=["Holdings_Tonnes"])
        logger.info(f"当前持有量: {len(result)} 个国家/机构")

        return result

    def build_model_features(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """
        从月度数据构建模型所需特征
        输出: 月度频率的央行购金特征
        """
        if monthly_data.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=monthly_data.index)

        # 1. 全球月度净购金量 (吨)
        features["cb_global_net_tonnes"] = monthly_data["Global_Total"]

        # 2. 3个月滚动购金量
        features["cb_global_3m_rolling"] = monthly_data["Global_Total"].rolling(3).sum()

        # 3. 6个月滚动购金量
        features["cb_global_6m_rolling"] = monthly_data["Global_Total"].rolling(6).sum()

        # 4. 12个月滚动购金量（年化）
        features["cb_global_12m_rolling"] = monthly_data["Global_Total"].rolling(12).sum()

        # 5. 购金量同比变化
        features["cb_global_yoy_change"] = monthly_data["Global_Total"] - \
            monthly_data["Global_Total"].shift(12)

        # 6. 主要购金央行合计
        if "Key_CBs_Total" in monthly_data.columns:
            features["cb_key_banks_net_tonnes"] = monthly_data["Key_CBs_Total"]

        # 7. 中国购金（单独列出，因为影响力最大）
        china_col = "China, P.R.: Mainland"
        if china_col in monthly_data.columns:
            features["cb_china_net_tonnes"] = monthly_data[china_col]

        features.index.name = "Date"
        return features

    def save_data(self, monthly: pd.DataFrame, annual: pd.DataFrame,
                  features: pd.DataFrame) -> list[str]:
        """保存央行购金数据"""
        cb_dir = os.path.join(self.raw_data_path, "central_bank")
        os.makedirs(cb_dir, exist_ok=True)

        saved = []

        if not monthly.empty:
            # 只保存全球总量和主要国家
            key_cols = ["Global_Total", "Key_CBs_Total"]
            key_cols += [c for c in KEY_CENTRAL_BANKS if c in monthly.columns]
            path = os.path.join(cb_dir, "monthly_changes.csv")
            monthly[key_cols].to_csv(path)
            saved.append(path)
            logger.info(f"已保存月度数据 -> {path}")

        if not annual.empty:
            path = os.path.join(cb_dir, "annual_changes.csv")
            annual[["Global_Total"]].to_csv(path)
            saved.append(path)
            logger.info(f"已保存年度数据 -> {path}")

        if not features.empty:
            path = os.path.join(cb_dir, "cb_features.csv")
            features.to_csv(path)
            saved.append(path)
            logger.info(f"已保存特征数据 -> {path}")

        return saved


def run_central_bank_gold_collection():
    """执行央行购金数据处理"""
    collector = CentralBankGoldCollector()

    print("=" * 60)
    print("处理央行购金数据 (WGC Excel)")
    print("=" * 60)

    monthly = collector.parse_monthly_changes()
    annual = collector.parse_annual_changes()
    features = collector.build_model_features(monthly)

    if monthly.empty:
        print("\n月度数据处理失败")
        return

    collector.save_data(monthly, annual, features)

    print(f"\n央行购金数据处理完成:")
    print(f"  月度数据: {len(monthly)} 个月 "
          f"({monthly.index[0].strftime('%Y-%m')} ~ {monthly.index[-1].strftime('%Y-%m')})")
    if not annual.empty:
        print(f"  年度数据: {len(annual)} 年")
    print(f"  模型特征: {len(features)} 行 x {len(features.columns)} 列")
    print(f"\n最近6个月全球央行净购金(吨):")
    for d, v in monthly["Global_Total"].tail(6).items():
        print(f"  {d.strftime('%Y-%m')}: {v:+.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_central_bank_gold_collection()
