"""解析 WGC (World Gold Council) 下载的 Excel 数据.

从 ~/Gold/data/download/ 读取, 输出到 ~/Gold/data/raw/ 各子目录.

用法:
    python scripts/parse_wgc_data.py

数据文件:
  - Changes_latest_*.xlsx → 央行储备月度变化
  - ETF_Flows_*.xlsx → 黄金ETF持仓和流量
  - gold-premiums.xlsx → 中国/印度黄金溢价
  - Prices.xlsx → 多币种金价
  - World_official_gold_holdings_*.xlsx → 全球储备排名
"""

import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Gold", "data", "download"))

DATA_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Gold", "data"))


def find_file(pattern):
    """在下载目录找匹配的文件."""
    for f in os.listdir(DL_DIR):
        if pattern.lower() in f.lower() and f.endswith(".xlsx"):
            return os.path.join(DL_DIR, f)
    return None


def parse_cb_changes():
    """解析央行储备月度变化 → monthly_changes.csv + cb_features.csv."""
    path = find_file("changes_latest") or find_file("changes")
    if not path:
        logger.warning("央行变化文件未找到")
        return

    logger.info(f"解析: {os.path.basename(path)}")
    df = pd.read_excel(path, sheet_name="Monthly", header=None)

    # Row 7 = header (Country Lookup | Country | Comments | date1 | date2 ...)
    # Row 8+ = data
    header_row = df.iloc[7]
    data = df.iloc[8:].copy()
    data.columns = header_row.values

    country_col = "Country"
    from datetime import datetime
    date_cols = [c for c in data.columns[3:]
                  if isinstance(c, (pd.Timestamp, datetime))]

    if not date_cols:
        logger.warning("未找到日期列")
        return

    df_data = data[[country_col] + date_cols].set_index(country_col)
    df_data = df_data.apply(pd.to_numeric, errors="coerce")
    monthly = df_data.T
    monthly.index = pd.to_datetime(monthly.index)
    monthly.index.name = "Date"

    # 计算全球合计
    monthly["Global_Total"] = monthly.sum(axis=1)

    # 主要央行
    key_banks = ["China, P.R.: Mainland", "Russian Federation", "India",
                  "Poland, Rep. of", "Singapore", "Kazakhstan, Rep. of"]
    key_cols = [c for c in key_banks if c in monthly.columns]
    if key_cols:
        monthly["Key_CBs_Total"] = monthly[key_cols].sum(axis=1)

    out_path = os.path.join(DATA_ROOT, "raw", "central_bank", "monthly_changes.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    monthly.to_csv(out_path)
    logger.info(f"央行月度变化: {len(monthly)} 行, {monthly.index[0]} ~ {monthly.index[-1]} → {out_path}")

    # 计算特征
    feat = pd.DataFrame(index=monthly.index)
    total = monthly["Global_Total"]
    feat["cb_global_net_tonnes"] = total
    feat["cb_global_3m_rolling"] = total.rolling(3).sum()
    feat["cb_global_6m_rolling"] = total.rolling(6).sum()
    feat["cb_global_12m_rolling"] = total.rolling(12).sum()
    feat["cb_global_yoy_change"] = total.diff(12)
    if "Key_CBs_Total" in monthly.columns:
        feat["cb_key_banks_net_tonnes"] = monthly["Key_CBs_Total"]
    if "China, P.R.: Mainland" in monthly.columns:
        feat["cb_china_net_tonnes"] = monthly["China, P.R.: Mainland"]

    feat_path = os.path.join(DATA_ROOT, "raw", "central_bank", "cb_features.csv")
    feat.to_csv(feat_path)
    logger.info(f"央行特征: {feat.shape[1]} 列 → {feat_path}")


def parse_etf_flows():
    """解析黄金ETF持仓和流量."""
    path = find_file("etf_flows") or find_file("ETF")
    if not path:
        logger.warning("ETF文件未找到")
        return

    logger.info(f"解析: {os.path.basename(path)}")

    # Holdings by month
    try:
        holdings = pd.read_excel(path, sheet_name="Holdings by month", header=1)
        # 找到数据起始
        date_col = None
        for c in holdings.columns:
            if "date" in str(c).lower() or isinstance(holdings[c].iloc[0], pd.Timestamp):
                date_col = c
                break

        if date_col:
            holdings = holdings.set_index(date_col)
            out = os.path.join(DATA_ROOT, "raw", "market", "gold_etf_holdings.csv")
            holdings.to_csv(out)
            logger.info(f"ETF 持仓: {len(holdings)} 行 → {out}")
        else:
            # 尝试另一种格式
            holdings = pd.read_excel(path, sheet_name="Holdings by month", header=2)
            out = os.path.join(DATA_ROOT, "raw", "market", "gold_etf_holdings.csv")
            holdings.to_csv(out)
            logger.info(f"ETF 持仓: {len(holdings)} 行 → {out}")
    except Exception as e:
        logger.warning(f"ETF Holdings 解析失败: {e}")

    # Demand by month (流入流出)
    try:
        demand = pd.read_excel(path, sheet_name="Demand by month", header=1)
        out = os.path.join(DATA_ROOT, "raw", "market", "gold_etf_demand.csv")
        demand.to_csv(out)
        logger.info(f"ETF 需求: {len(demand)} 行 → {out}")
    except Exception as e:
        logger.warning(f"ETF Demand 解析失败: {e}")


def parse_premiums():
    """解析中国/印度黄金溢价."""
    path = find_file("premium")
    if not path:
        logger.warning("溢价文件未找到")
        return

    logger.info(f"解析: {os.path.basename(path)}")

    for sheet, name in [("Chinese premiums-discounts", "china"),
                          ("Indian premiums-discounts", "india")]:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=4)
            df.columns = ["Date", "Premium_USD"]
            df = df.dropna(subset=["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            out = os.path.join(DATA_ROOT, "raw", "market", f"gold_premium_{name}.csv")
            df.to_csv(out)
            logger.info(f"{name} 溢价: {len(df)} 行, ~ {df.index[-1].date()} → {out}")
        except Exception as e:
            logger.warning(f"{name} 溢价解析失败: {e}")


def parse_prices():
    """解析多币种金价."""
    path = find_file("prices")
    if not path:
        logger.warning("价格文件未找到")
        return

    logger.info(f"解析: {os.path.basename(path)}")

    for sheet, freq in [("Monthly_Avg", "monthly"), ("Quarterly_Avg", "quarterly")]:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=5)
            # 第一列是日期, 后面是各币种
            df.columns = ["Date"] + list(df.columns[1:])
            df = df.dropna(subset=["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            out = os.path.join(DATA_ROOT, "raw", "market", f"gold_price_{freq}.csv")
            df.to_csv(out)
            logger.info(f"金价 ({freq}): {len(df)} 行 → {out}")
        except Exception as e:
            logger.warning(f"金价 {freq} 解析失败: {e}")


def main():
    if not os.path.isdir(DL_DIR):
        logger.error(f"下载目录不存在: {DL_DIR}")
        return

    logger.info(f"WGC 数据解析: {DL_DIR}")

    parse_cb_changes()
    parse_etf_flows()
    parse_premiums()
    parse_prices()

    logger.info("完成!")


if __name__ == "__main__":
    main()
