"""
宏观经济数据采集模块
通过 FRED API 获取美债实际收益率、通胀预期、美元指数、联邦债务等宏观数据
"""

import os
import logging
from datetime import datetime

import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class MacroDataCollector:
    """FRED 宏观经济数据采集器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        os.makedirs(self.raw_data_path, exist_ok=True)

        self.api_key = self.config.get("fred_api_key", "")
        self.start_date = self.config["data"]["start_date"]
        self.end_date = self.config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")

        self.fred_series = self.config.get("fred", {})

        self._fred = None

    def _get_fred_client(self):
        """获取 FRED API 客户端"""
        if self._fred is None:
            if not self.api_key or self.api_key == "YOUR_FRED_API_KEY":
                raise ValueError(
                    "请先设置 FRED API Key!\n"
                    "1. 访问 https://fred.stlouisfed.org/docs/api/api_key.html 免费申请\n"
                    "2. 将 API Key 填入 config/settings.yaml 的 fred_api_key 字段"
                )
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def fetch_single_series(self, name: str, series_id: str,
                            start: str = None, end: str = None) -> pd.Series:
        """获取单个 FRED 数据序列"""
        start = start or self.start_date
        end = end or self.end_date

        logger.info(f"正在获取 FRED {name} ({series_id}): {start} ~ {end}")

        try:
            fred = self._get_fred_client()
            data = fred.get_series(series_id, observation_start=start, observation_end=end)

            if data.empty:
                logger.warning(f"{name} ({series_id}) 返回空数据")
                return pd.Series(dtype=float)

            data.index.name = "Date"
            data.name = series_id

            # 统计有效数据
            valid_count = data.notna().sum()
            logger.info(f"{name}: 获取 {len(data)} 条记录 (有效 {valid_count}), "
                        f"时间范围 {data.index[0].strftime('%Y-%m-%d')} ~ "
                        f"{data.index[-1].strftime('%Y-%m-%d')}")
            return data

        except ValueError:
            raise  # API Key 错误直接抛出
        except Exception as e:
            logger.error(f"获取 {name} ({series_id}) 失败: {e}")
            return pd.Series(dtype=float)

    def fetch_all_macro_data(self) -> dict[str, pd.Series]:
        """获取所有 FRED 宏观数据"""
        results = {}

        for name, info in self.fred_series.items():
            series_id = info["series_id"]
            data = self.fetch_single_series(name, series_id)
            if not data.empty:
                results[name] = data

        return results

    def save_macro_data(self, data: dict[str, pd.Series]) -> dict[str, str]:
        """保存宏观数据到 CSV"""
        macro_dir = os.path.join(self.raw_data_path, "macro")
        os.makedirs(macro_dir, exist_ok=True)

        saved_paths = {}
        for name, series in data.items():
            filepath = os.path.join(macro_dir, f"{name}.csv")
            df = series.to_frame()
            df.to_csv(filepath)
            saved_paths[name] = filepath
            logger.info(f"已保存 {name} -> {filepath} ({len(series)} 行)")

        return saved_paths

    def build_macro_panel(self, data: dict[str, pd.Series]) -> pd.DataFrame:
        """
        将所有宏观数据合并为一个面板 DataFrame
        不同频率的数据用前向填充对齐到日频
        """
        if not data:
            return pd.DataFrame()

        # 将所有序列合并
        frames = {}
        for name, series in data.items():
            series_id = self.fred_series[name]["series_id"]
            frames[series_id] = series

        panel = pd.DataFrame(frames)
        panel.index.name = "Date"

        # 按日期排序，前向填充（低频数据在高频时间轴上填充）
        panel = panel.sort_index()
        panel = panel.ffill()

        logger.info(f"宏观数据面板: {panel.shape[0]} 行 x {panel.shape[1]} 列")
        logger.info(f"列: {list(panel.columns)}")

        return panel


def run_macro_data_collection():
    """执行完整的宏观数据采集"""
    collector = MacroDataCollector()

    print("=" * 60)
    print("开始采集宏观经济数据 (FRED)")
    print("=" * 60)

    try:
        macro_data = collector.fetch_all_macro_data()
    except ValueError as e:
        print(f"\n错误: {e}")
        return None, None

    saved = collector.save_macro_data(macro_data)

    print(f"\n宏观数据采集完成，共 {len(saved)} 个数据序列:")
    for name, path in saved.items():
        series_id = collector.fred_series[name]["series_id"]
        freq = collector.fred_series[name]["frequency"]
        count = len(macro_data[name])
        print(f"  {name} ({series_id}): {count} 条记录, 频率={freq}")

    # 构建合并面板
    panel = collector.build_macro_panel(macro_data)
    if not panel.empty:
        panel_path = os.path.join(collector.raw_data_path, "macro", "macro_panel.csv")
        panel.to_csv(panel_path)
        print(f"\n合并面板已保存: {panel_path}")
        print(f"面板维度: {panel.shape[0]} 行 x {panel.shape[1]} 列")

    return macro_data, panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_macro_data_collection()
