"""
波动率数据采集模块
获取 GVZ (黄金波动率指数)、VIX 期限结构、MOVE 指数等波动率相关数据
"""

import os
import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class VolDataCollector:
    """波动率数据采集器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        os.makedirs(self.raw_data_path, exist_ok=True)

    def fetch_gvz(self) -> pd.DataFrame:
        """
        获取 CBOE GVZ (Gold ETF Volatility Index) 历史数据
        GVZ 是基于 GLD 期权价格构造的黄金隐含波动率指数
        """
        url = self.config["cboe"]["gvz"]["url"]
        logger.info(f"正在从 CBOE 获取 GVZ 数据: {url}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            # CBOE CSV 格式可能有变化，尝试常见的列名
            date_col = None
            close_col = None
            for col in df.columns:
                col_lower = col.strip().lower()
                if "date" in col_lower:
                    date_col = col
                elif "close" in col_lower:
                    close_col = col

            if date_col is None or close_col is None:
                logger.warning(f"GVZ CSV 列名不匹配，实际列: {list(df.columns)}")
                # 尝试用位置匹配
                if len(df.columns) >= 2:
                    date_col = df.columns[0]
                    close_col = df.columns[-1]
                    logger.info(f"尝试使用位置匹配: date={date_col}, close={close_col}")
                else:
                    return pd.DataFrame()

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = "Date"
            df = df.sort_index()

            logger.info(f"GVZ: 获取 {len(df)} 条记录, "
                        f"时间范围 {df.index[0].strftime('%Y-%m-%d')} ~ "
                        f"{df.index[-1].strftime('%Y-%m-%d')}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"获取 GVZ 数据失败 (网络错误): {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"解析 GVZ 数据失败: {e}")
            return pd.DataFrame()

    def fetch_vix_term_structure(self) -> pd.DataFrame:
        """
        获取 VIX 期限结构相关数据
        通过 Yahoo Finance 获取不同期限的 VIX 期货
        """
        import yfinance as yf

        logger.info("正在获取 VIX 期限结构数据...")

        # VIX 和 VIX 相关 ETF/ETN 来近似期限结构
        vix_tickers = {
            "VIX": "^VIX",          # VIX 现货
            "VIX9D": "^VIX9D",      # 9天VIX
            "VIX3M": "^VIX3M",      # 3个月VIX
            "VIX6M": "^VIX6M",      # 6个月VIX
        }

        results = {}
        start = self.config["data"]["start_date"]
        end = self.config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")

        for name, ticker in vix_tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    results[name] = data["Close"]
                    logger.info(f"  {name}: {len(data)} 条记录")
                else:
                    logger.warning(f"  {name} ({ticker}) 返回空数据")
            except Exception as e:
                logger.warning(f"  获取 {name} ({ticker}) 失败: {e}")

        if results:
            df = pd.DataFrame(results)
            df.index.name = "Date"
            return df
        return pd.DataFrame()

    def fetch_move_index(self) -> pd.Series:
        """
        获取 MOVE 指数 (ICE BofA MOVE Index，债券波动率指数)
        FRED series: MOVE (可能不可用，备用方案用 Yahoo Finance)
        """
        import yfinance as yf

        logger.info("正在获取 MOVE 指数...")

        # 先尝试 FRED
        try:
            api_key = self.config.get("fred_api_key", "")
            if api_key and api_key != "YOUR_FRED_API_KEY":
                from fredapi import Fred
                fred = Fred(api_key=api_key)
                data = fred.get_series("MOVE",
                                       observation_start=self.config["data"]["start_date"])
                if not data.empty:
                    data.name = "MOVE"
                    logger.info(f"MOVE (FRED): {len(data)} 条记录")
                    return data
        except Exception as e:
            logger.info(f"FRED MOVE 不可用 ({e})，尝试备用方案...")

        # 备用: 尝试 Yahoo Finance 的 ^MOVE
        try:
            start = self.config["data"]["start_date"]
            end = self.config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")
            data = yf.download("^MOVE", start=start, end=end, auto_adjust=True, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                series = data["Close"]
                series.name = "MOVE"
                logger.info(f"MOVE (Yahoo): {len(data)} 条记录")
                return series
        except Exception as e:
            logger.warning(f"Yahoo MOVE 也不可用: {e}")

        logger.warning("MOVE 指数暂时无法获取，后续可通过付费数据源补充")
        return pd.Series(dtype=float)

    def save_vol_data(self, gvz: pd.DataFrame, vix_term: pd.DataFrame,
                      move: pd.Series) -> list[str]:
        """保存波动率数据"""
        vol_dir = os.path.join(self.raw_data_path, "volatility")
        os.makedirs(vol_dir, exist_ok=True)

        saved = []

        if not gvz.empty:
            path = os.path.join(vol_dir, "gvz.csv")
            gvz.to_csv(path)
            saved.append(path)
            logger.info(f"已保存 GVZ -> {path}")

        if not vix_term.empty:
            path = os.path.join(vol_dir, "vix_term_structure.csv")
            vix_term.to_csv(path)
            saved.append(path)
            logger.info(f"已保存 VIX期限结构 -> {path}")

        if not move.empty:
            path = os.path.join(vol_dir, "move.csv")
            move.to_frame().to_csv(path)
            saved.append(path)
            logger.info(f"已保存 MOVE -> {path}")

        return saved


def run_vol_data_collection():
    """执行波动率数据采集"""
    collector = VolDataCollector()

    print("=" * 60)
    print("开始采集波动率数据")
    print("=" * 60)

    gvz = collector.fetch_gvz()
    vix_term = collector.fetch_vix_term_structure()
    move = collector.fetch_move_index()

    saved = collector.save_vol_data(gvz, vix_term, move)

    print(f"\n波动率数据采集完成:")
    if not gvz.empty:
        print(f"  GVZ: {len(gvz)} 条记录")
    else:
        print(f"  GVZ: 获取失败（后续排查）")

    if not vix_term.empty:
        print(f"  VIX期限结构: {len(vix_term)} 条记录, 列={list(vix_term.columns)}")
    else:
        print(f"  VIX期限结构: 获取失败")

    if not move.empty:
        print(f"  MOVE: {len(move)} 条记录")
    else:
        print(f"  MOVE: 暂时不可用")

    return gvz, vix_term, move


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_vol_data_collection()
