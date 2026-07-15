"""
经济事件日历模块
获取FOMC会议日期、非农就业、CPI发布日等重要经济事件
以及CME FedWatch降息/加息概率
"""

import os
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# FOMC 会议决议日（第二天，即声明发布日）
# 来源: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# 硬编码比爬取更可靠，每年更新一次即可
FOMC_DECISION_DATES = [
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]


class EconomicEventsCollector:
    """经济事件日历采集器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        self.api_key = self.config.get("fred_api_key", "")

    def fetch_fomc_dates(self) -> pd.DataFrame:
        """
        获取FOMC会议决议日（声明发布日）
        使用硬编码数据，比爬取更可靠
        """
        logger.info("加载 FOMC 会议日期（硬编码）...")

        fomc_dates = pd.to_datetime(FOMC_DECISION_DATES)
        df = pd.DataFrame({"Date": fomc_dates, "Event": "FOMC"})
        df = df.sort_values("Date").reset_index(drop=True)

        logger.info(f"FOMC 会议日期: {len(df)} 个 "
                    f"({df['Date'].iloc[0].strftime('%Y-%m-%d')} ~ "
                    f"{df['Date'].iloc[-1].strftime('%Y-%m-%d')})")
        return df

    def fetch_fred_release_dates(self, series_id: str, name: str) -> pd.DataFrame:
        """
        从FRED获取某个数据系列的发布日期
        这些发布日期就是市场事件日
        """
        if not self.api_key or self.api_key == "YOUR_FRED_API_KEY":
            return pd.DataFrame()

        logger.info(f"正在获取 {name} 发布日期 ({series_id})...")

        try:
            # 使用FRED API的 release/dates 端点
            # 先获取series对应的release ID
            url = f"https://api.stlouisfed.org/fred/series/release"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            release_data = resp.json()

            if "releases" not in release_data or not release_data["releases"]:
                logger.warning(f"  {series_id} 没有对应的release信息")
                return pd.DataFrame()

            release_id = release_data["releases"][0]["id"]

            # 获取该release的发布日期
            url2 = f"https://api.stlouisfed.org/fred/release/dates"
            params2 = {
                "release_id": release_id,
                "api_key": self.api_key,
                "file_type": "json",
                "include_release_dates_with_no_data": "false",
                "sort_order": "asc",
            }
            resp2 = requests.get(url2, params=params2, timeout=15)
            resp2.raise_for_status()
            dates_data = resp2.json()

            if "release_dates" not in dates_data:
                return pd.DataFrame()

            dates = [rd["date"] for rd in dates_data["release_dates"]]
            df = pd.DataFrame({"Date": pd.to_datetime(dates), "Event": name})

            logger.info(f"  {name}: {len(df)} 个发布日期")
            return df

        except Exception as e:
            logger.warning(f"  获取 {name} 发布日期失败: {e}")
            return pd.DataFrame()

    def fetch_all_economic_events(self) -> pd.DataFrame:
        """获取所有重要经济事件日历"""
        all_events = []

        # 1. FOMC 会议
        fomc = self.fetch_fomc_dates()
        if not fomc.empty:
            all_events.append(fomc)

        # 2. 非农就业报告 (Employment Situation)
        nfp = self.fetch_fred_release_dates("PAYEMS", "NFP")
        if not nfp.empty:
            all_events.append(nfp)

        # 3. CPI 消费者物价指数
        cpi = self.fetch_fred_release_dates("CPIAUCSL", "CPI")
        if not cpi.empty:
            all_events.append(cpi)

        # 4. PPI 生产者物价指数
        ppi = self.fetch_fred_release_dates("PPIACO", "PPI")
        if not ppi.empty:
            all_events.append(ppi)

        # 5. GDP
        gdp = self.fetch_fred_release_dates("GDP", "GDP")
        if not gdp.empty:
            all_events.append(gdp)

        # 6. ISM 制造业PMI
        ism = self.fetch_fred_release_dates("MANEMP", "ISM_Manufacturing")
        if not ism.empty:
            all_events.append(ism)

        if not all_events:
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        combined = combined.sort_values("Date").reset_index(drop=True)

        logger.info(f"经济事件日历合计: {len(combined)} 条")
        return combined

    def build_event_features(self, events: pd.DataFrame,
                             date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        将事件日历转换为模型特征
        对每个交易日生成:
        - 距离下一个FOMC的天数
        - 距离下一个NFP的天数
        - 距离下一个CPI的天数
        - 当天是否是事件日 (one-hot)
        - 事件前N天窗口标记
        """
        if events.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=date_index)
        features.index.name = "Date"

        event_types = events["Event"].unique()

        for event_type in event_types:
            event_dates = events[events["Event"] == event_type]["Date"].values

            # 1. 距离下一个事件的天数
            col_days = f"days_to_next_{event_type}"
            features[col_days] = None

            for i, date in enumerate(date_index):
                future_events = event_dates[event_dates > date]
                if len(future_events) > 0:
                    days = (pd.Timestamp(future_events[0]) - date).days
                    features.loc[date, col_days] = days

            features[col_days] = pd.to_numeric(features[col_days], errors="coerce")

            # 2. 当天是否是事件日
            col_is = f"is_{event_type}_day"
            features[col_is] = date_index.isin(event_dates).astype(int)

            # 3. 事件前3天窗口
            col_window = f"{event_type}_window_3d"
            features[col_window] = 0
            for ed in event_dates:
                mask = (date_index >= pd.Timestamp(ed) - timedelta(days=3)) & \
                       (date_index <= pd.Timestamp(ed))
                features.loc[mask, col_window] = 1

        return features

    def save_events(self, events: pd.DataFrame, features: pd.DataFrame = None) -> list[str]:
        """保存事件数据"""
        events_dir = os.path.join(self.raw_data_path, "events")
        os.makedirs(events_dir, exist_ok=True)

        saved = []

        path = os.path.join(events_dir, "economic_calendar.csv")
        events.to_csv(path, index=False)
        saved.append(path)
        logger.info(f"已保存经济事件日历 -> {path} ({len(events)} 条)")

        if features is not None and not features.empty:
            path = os.path.join(events_dir, "event_features.csv")
            features.to_csv(path)
            saved.append(path)
            logger.info(f"已保存事件特征 -> {path}")

        return saved


def run_economic_events_collection():
    """执行经济事件采集"""
    collector = EconomicEventsCollector()

    print("=" * 60)
    print("采集经济事件日历")
    print("=" * 60)

    events = collector.fetch_all_economic_events()

    if events.empty:
        print("\n经济事件采集失败")
        return pd.DataFrame()

    collector.save_events(events)

    print(f"\n经济事件日历采集完成:")
    for event_type in events["Event"].unique():
        count = len(events[events["Event"] == event_type])
        subset = events[events["Event"] == event_type]
        print(f"  {event_type}: {count} 条 "
              f"({subset['Date'].iloc[0].strftime('%Y-%m-%d')} ~ "
              f"{subset['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    return events


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_economic_events_collection()
