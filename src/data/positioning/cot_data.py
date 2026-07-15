"""
CFTC COT (Commitments of Traders) 数据采集模块
获取 COMEX 黄金期货的持仓数据，包括商业/非商业/基金持仓
"""

import os
import logging
import zipfile
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
import requests

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# CFTC 黄金期货匹配规则
GOLD_MARKET_NAME = "GOLD - COMMODITY EXCHANGE INC."


class COTDataCollector:
    """CFTC COT 报告数据采集器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        os.makedirs(self.raw_data_path, exist_ok=True)

    def fetch_cot_historical(self, start_year: int = 2006) -> pd.DataFrame:
        """
        获取 CFTC COT 历史数据 (Futures Only, Legacy Report)
        CFTC 提供年度压缩文件下载

        数据包含:
        - 商业持仓 (Commercial): 生产商/消费商的套期保值
        - 非商业持仓 (Non-Commercial): 基金/投机者
        - 非报告持仓 (Nonreportable)
        """
        current_year = datetime.now().year
        all_data = []

        for year in range(start_year, current_year + 1):
            df = self._fetch_cot_year(year)
            if df is not None and not df.empty:
                all_data.append(df)

        if not all_data:
            logger.error("未能获取任何 COT 数据")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values("Date").reset_index(drop=True)

        logger.info(f"COT 黄金数据合计: {len(combined)} 条周报记录, "
                    f"{combined['Date'].iloc[0].strftime('%Y-%m-%d')} ~ "
                    f"{combined['Date'].iloc[-1].strftime('%Y-%m-%d')}")

        return combined

    def _fetch_cot_year(self, year: int) -> pd.DataFrame:
        """获取某一年的 COT 数据"""
        # CFTC 年度文件 URL (Futures Only, Legacy)
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"

        logger.info(f"正在获取 {year} 年 COT 数据: {url}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            # 解压 ZIP 文件
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                # ZIP 里通常只有一个文件
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as f:
                    content = f.read().decode("utf-8", errors="replace")
                    df = pd.read_csv(StringIO(content), low_memory=False)

            # 筛选黄金相关记录
            gold_df = self._filter_gold_records(df)
            if gold_df is not None:
                logger.info(f"  {year}: {len(gold_df)} 条黄金记录")
            return gold_df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.info(f"  {year} 年数据文件不存在 (可能还未发布)")
            else:
                logger.warning(f"  获取 {year} 年数据失败: {e}")
            return None
        except Exception as e:
            logger.warning(f"  解析 {year} 年数据失败: {e}")
            return None

    def _filter_gold_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 COT 原始数据中筛选黄金记录并提取关键字段"""
        df.columns = df.columns.str.strip()

        # 精确匹配 COMEX 黄金 (排除 MICRO GOLD 等)
        name_col = "Market and Exchange Names"
        if name_col not in df.columns:
            # 兼容旧格式（下划线分隔）
            name_col = "Market_and_Exchange_Names"

        if name_col not in df.columns:
            logger.warning(f"未找到市场名称列，实际列: {list(df.columns[:5])}")
            return None

        gold_mask = df[name_col].str.strip() == GOLD_MARKET_NAME
        if gold_mask.sum() == 0:
            # 回退到模糊匹配，但排除 MICRO
            gold_mask = (
                df[name_col].str.contains("GOLD", case=False, na=False)
                & df[name_col].str.contains("COMMODITY EXCHANGE", case=False, na=False)
                & ~df[name_col].str.contains("MICRO", case=False, na=False)
                & ~df[name_col].str.contains("COINBASE", case=False, na=False)
            )

        if gold_mask.sum() == 0:
            logger.warning("未找到黄金相关记录")
            return None

        gold_df = df[gold_mask].copy()

        # CFTC CSV 有两种列名风格：
        # 新格式: "As of Date in Form YYYY-MM-DD" (空格分隔)
        # 旧格式: "As_of_Date_In_Form_YYMMDD" (下划线分隔)
        column_mapping = {
            # 日期
            "As of Date in Form YYYY-MM-DD": "Date",
            "As_of_Date_In_Form_YYMMDD": "Date_YYMMDD",
            # 非商业 (投机者/基金) 持仓
            "Noncommercial Positions-Long (All)": "NonComm_Long",
            "Noncommercial Positions-Short (All)": "NonComm_Short",
            "Noncommercial Positions-Spreading (All)": "NonComm_Spread",
            # 商业 (套期保值) 持仓
            "Commercial Positions-Long (All)": "Comm_Long",
            "Commercial Positions-Short (All)": "Comm_Short",
            # 总持仓
            "Open Interest (All)": "Open_Interest",
            # 变化量
            "Change in Open Interest (All)": "OI_Change",
            "Change in Noncommercial-Long (All)": "NonComm_Long_Change",
            "Change in Noncommercial-Short (All)": "NonComm_Short_Change",
            "Change in Commercial-Long (All)": "Comm_Long_Change",
            "Change in Commercial-Short (All)": "Comm_Short_Change",
        }

        result = pd.DataFrame()
        for orig_col, new_col in column_mapping.items():
            if orig_col in gold_df.columns:
                result[new_col] = gold_df[orig_col].values

        # 解析日期 - 优先用 YYYY-MM-DD 格式
        if "Date" in result.columns:
            result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
        elif "Date_YYMMDD" in result.columns:
            result["Date"] = pd.to_datetime(result["Date_YYMMDD"], format="%y%m%d", errors="coerce")

        if "Date_YYMMDD" in result.columns:
            result = result.drop(columns=["Date_YYMMDD"])

        result = result.dropna(subset=["Date"])

        # 计算衍生指标
        if "NonComm_Long" in result.columns and "NonComm_Short" in result.columns:
            result["NonComm_Net"] = result["NonComm_Long"] - result["NonComm_Short"]
        if "Comm_Long" in result.columns and "Comm_Short" in result.columns:
            result["Comm_Net"] = result["Comm_Long"] - result["Comm_Short"]

        return result

    def save_cot_data(self, cot_data: pd.DataFrame) -> str:
        """保存 COT 数据"""
        cot_dir = os.path.join(self.raw_data_path, "cot")
        os.makedirs(cot_dir, exist_ok=True)

        filepath = os.path.join(cot_dir, "gold_cot.csv")
        cot_data.to_csv(filepath, index=False)
        logger.info(f"已保存 COT 数据 -> {filepath} ({len(cot_data)} 行)")
        return filepath


def run_cot_data_collection():
    """执行 COT 数据采集"""
    collector = COTDataCollector()

    print("=" * 60)
    print("开始采集 CFTC COT 持仓数据")
    print("=" * 60)

    cot_data = collector.fetch_cot_historical(start_year=2006)

    if cot_data.empty:
        print("\nCOT 数据采集失败")
        return pd.DataFrame()

    collector.save_cot_data(cot_data)

    print(f"\nCOT 数据采集完成:")
    print(f"  记录数: {len(cot_data)}")
    print(f"  时间范围: {cot_data['Date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{cot_data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    if "NonComm_Net" in cot_data.columns:
        print(f"  非商业净持仓 (最新): {cot_data['NonComm_Net'].iloc[-1]:,.0f}")

    return cot_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_cot_data_collection()
