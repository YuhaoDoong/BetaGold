"""央行购金数据自动获取.

来源:
  1. IMF IFS (International Financial Statistics) — 月度, API 可用
  2. World Gold Council — 需要手动下载 Excel (备用)

用法:
    python scripts/fetch_central_bank.py           # 尝试在线更新
    python scripts/fetch_central_bank.py --excel FILE  # 从 WGC Excel 导入

输出: data/raw/central_bank/monthly_changes.csv + cb_features.csv
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "..", "Gold", "data", "raw", "central_bank")

# IMF IFS API — 黄金储备 (盎司)
# 国际货币基金组织, 免费 API
IMF_IFS_URL = "https://www.imf.org/external/datamapper/api/v1/GOLD"

# 主要购金国
KEY_COUNTRIES = {
    "CHN": "China, P.R.: Mainland",
    "RUS": "Russian Federation",
    "IND": "India",
    "POL": "Poland, Rep. of",
    "SGP": "Singapore",
    "KAZ": "Kazakhstan, Rep. of",
    "UZB": "Uzbekistan, Rep. of",
    "QAT": "Qatar",
}


def fetch_imf_gold_reserves():
    """从 IMF API 获取各国黄金储备数据."""
    try:
        resp = requests.get(IMF_IFS_URL, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("IMF gold reserves data fetched")
            return data
    except Exception as e:
        logger.warning(f"IMF API failed: {e}")

    # 备用: IMF Data API v2
    try:
        url = ("https://sdmxcentral.imf.org/ws/public/sdmxapi/rest/data/"
               "IFS/M..RAXG_USD..?format=csv&startPeriod=2020")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(pd.io.common.StringIO(resp.text))
            logger.info(f"IMF SDMX: {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"IMF SDMX failed: {e}")

    return None


def parse_wgc_excel(excel_path):
    """解析 World Gold Council 下载的 Excel 文件.

    文件格式: "Changes in World Official Gold Reserves"
    表头在第7行, 国家在第1列, 日期从第3列开始.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="Changes Monthly",
                           header=7, index_col=1)
        # 转置: 行=国家, 列=月份 → 行=月份, 列=国家
        df = df.iloc[:, 2:]  # 跳过前两列 (代码等)
        df.columns = pd.to_datetime(df.columns)
        df = df.T
        df.index.name = "Date"

        logger.info(f"WGC Excel parsed: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to parse WGC Excel: {e}")
        return None


def compute_cb_features(monthly_df):
    """从月度数据计算央行购金特征."""
    feat = pd.DataFrame(index=monthly_df.index)

    # 全球合计
    if "Global_Total" in monthly_df.columns:
        total = monthly_df["Global_Total"]
    else:
        total = monthly_df.sum(axis=1)

    feat["cb_global_net_tonnes"] = total
    feat["cb_global_3m_rolling"] = total.rolling(3).sum()
    feat["cb_global_6m_rolling"] = total.rolling(6).sum()
    feat["cb_global_12m_rolling"] = total.rolling(12).sum()
    feat["cb_global_yoy_change"] = total.diff(12)

    # 主要央行合计
    key_cols = [c for c in monthly_df.columns
                if any(k in c for k in KEY_COUNTRIES.values())]
    if key_cols:
        feat["cb_key_banks_net_tonnes"] = monthly_df[key_cols].sum(axis=1)

    # 中国
    china_cols = [c for c in monthly_df.columns if "China" in c]
    if china_cols:
        feat["cb_china_net_tonnes"] = monthly_df[china_cols[0]]

    return feat


def update_central_bank(excel_path=None):
    """更新央行购金数据."""
    os.makedirs(DATA_DIR, exist_ok=True)
    monthly_path = os.path.join(DATA_DIR, "monthly_changes.csv")
    features_path = os.path.join(DATA_DIR, "cb_features.csv")

    monthly = None

    if excel_path and os.path.exists(excel_path):
        # 从 WGC Excel 导入
        raw = parse_wgc_excel(excel_path)
        if raw is not None:
            monthly = raw
            monthly.to_csv(monthly_path)
            logger.info(f"Saved monthly data from Excel")

    if monthly is None:
        # 尝试 IMF API
        imf_data = fetch_imf_gold_reserves()
        if imf_data is not None:
            logger.info("IMF data available (需要进一步解析)")
            # IMF 数据格式复杂, 这里简化处理

    if monthly is None and os.path.exists(monthly_path):
        monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        logger.info(f"Using existing monthly data: {monthly.index[-1].date()}")

    if monthly is not None:
        features = compute_cb_features(monthly)
        features.to_csv(features_path)
        logger.info(f"CB features saved: {features.index[-1].date()} ({len(features)} rows)")
    else:
        logger.warning("No central bank data available")
        logger.info("请手动下载 WGC Excel:")
        logger.info("  https://china.gold.org/goldhub/data/gold-reserves-by-country")
        logger.info("  然后运行: python scripts/fetch_central_bank.py --excel <文件路径>")


def main():
    parser = argparse.ArgumentParser(description="央行购金数据更新")
    parser.add_argument("--excel", help="WGC Excel 文件路径")
    args = parser.parse_args()

    update_central_bank(excel_path=args.excel)


if __name__ == "__main__":
    main()
