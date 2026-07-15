"""
市场数据采集模块
通过 Yahoo Finance 获取 GLD、黄金期货、美元指数、VIX 等市场数据
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Yahoo Finance 市场数据采集器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        os.makedirs(self.raw_data_path, exist_ok=True)

        self.start_date = self.config["data"]["start_date"]
        self.end_date = self.config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")

        # 从配置中提取所有 Yahoo Finance ticker
        self.tickers = {}
        for key, val in self.config["yahoo_finance"].items():
            self.tickers[key] = val["ticker"]

    def fetch_single_ticker(self, name: str, ticker: str,
                            start: str = None, end: str = None) -> pd.DataFrame:
        """获取单个 ticker 的历史数据"""
        start = start or self.start_date
        end = end or self.end_date

        logger.info(f"正在获取 {name} ({ticker}) 数据: {start} ~ {end}")

        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

            if data.empty:
                logger.warning(f"{name} ({ticker}) 返回空数据")
                return pd.DataFrame()

            # yfinance 可能返回 MultiIndex columns，如果只有一个 ticker 则展平
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.index.name = "Date"
            logger.info(f"{name}: 获取 {len(data)} 条记录, "
                        f"时间范围 {data.index[0].strftime('%Y-%m-%d')} ~ "
                        f"{data.index[-1].strftime('%Y-%m-%d')}")
            return data

        except Exception as e:
            logger.error(f"获取 {name} ({ticker}) 失败: {e}")
            return pd.DataFrame()

    def fetch_all_market_data(self) -> dict[str, pd.DataFrame]:
        """获取所有市场数据"""
        results = {}

        for name, ticker in self.tickers.items():
            df = self.fetch_single_ticker(name, ticker)
            if not df.empty:
                results[name] = df

        return results

    def save_market_data(self, data: dict[str, pd.DataFrame]) -> dict[str, str]:
        """保存市场数据到 CSV"""
        saved_paths = {}
        market_dir = os.path.join(self.raw_data_path, "market")
        os.makedirs(market_dir, exist_ok=True)

        for name, df in data.items():
            filepath = os.path.join(market_dir, f"{name}.csv")
            df.to_csv(filepath)
            saved_paths[name] = filepath
            logger.info(f"已保存 {name} -> {filepath} ({len(df)} 行)")

        return saved_paths

    def fetch_gld_options_chain(self) -> dict:
        """获取 GLD 当前期权链数据（用于策略层，非历史数据）"""
        logger.info("正在获取 GLD 期权链...")

        try:
            gld = yf.Ticker("GLD")
            expirations = gld.options

            if not expirations:
                logger.warning("未获取到 GLD 期权到期日")
                return {}

            logger.info(f"GLD 可用期权到期日: {len(expirations)} 个")
            logger.info(f"最近到期日: {expirations[:5]}")

            options_data = {}
            for exp in expirations[:6]:  # 只取最近6个到期日，避免请求过多
                try:
                    chain = gld.option_chain(exp)
                    options_data[exp] = {
                        "calls": chain.calls,
                        "puts": chain.puts,
                    }
                    logger.info(f"  {exp}: {len(chain.calls)} calls, {len(chain.puts)} puts")
                except Exception as e:
                    logger.warning(f"  获取 {exp} 期权链失败: {e}")

            return options_data

        except Exception as e:
            logger.error(f"获取 GLD 期权链失败: {e}")
            return {}

    def save_options_chain(self, options_data: dict) -> list[str]:
        """保存期权链数据"""
        options_dir = os.path.join(self.raw_data_path, "options")
        os.makedirs(options_dir, exist_ok=True)

        saved = []
        today = datetime.now().strftime("%Y%m%d")

        for exp, chain in options_data.items():
            for opt_type in ["calls", "puts"]:
                filename = f"gld_{opt_type}_{exp}_{today}.csv"
                filepath = os.path.join(options_dir, filename)
                chain[opt_type].to_csv(filepath, index=False)
                saved.append(filepath)

        logger.info(f"已保存 {len(saved)} 个期权链文件到 {options_dir}")
        return saved


def run_market_data_collection():
    """执行完整的市场数据采集"""
    collector = MarketDataCollector()

    print("=" * 60)
    print("开始采集市场数据 (Yahoo Finance)")
    print("=" * 60)

    # 1. 采集所有市场行情数据
    market_data = collector.fetch_all_market_data()
    saved = collector.save_market_data(market_data)

    print(f"\n市场数据采集完成，共 {len(saved)} 个数据集:")
    for name, path in saved.items():
        df = market_data[name]
        print(f"  {name}: {len(df)} 条记录")

    # 2. 采集 GLD 期权链
    options = collector.fetch_gld_options_chain()
    if options:
        collector.save_options_chain(options)

    return market_data, options


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_market_data_collection()
