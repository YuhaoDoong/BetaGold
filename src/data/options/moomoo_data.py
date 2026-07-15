"""
Moomoo API 数据采集模块
通过 moomoo OpenD 获取 GLD 期权链详细数据，包括：
- 期权链 (所有到期日、行权价)
- 实时报价 + Greeks (Delta/Gamma/Vega/Theta/Rho)
- 隐含波动率 (IV)
- 未平仓合约数 (Open Interest)
- 历史K线

前置条件:
1. 安装 moomoo-api: pip install moomoo-api
2. 启动 OpenD 客户端 (默认端口 11111)
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# GLD 在 moomoo 中的代码
GLD_CODE = "US.GLD"


class MoomooDataCollector:
    """Moomoo API 数据采集器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, config: dict = None):
        self.host = host
        self.port = port
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        self._quote_ctx = None

    def _get_quote_ctx(self):
        """获取行情上下文（延迟初始化）"""
        if self._quote_ctx is None:
            from moomoo import OpenQuoteContext
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            logger.info(f"已连接 Moomoo OpenD ({self.host}:{self.port})")
        return self._quote_ctx

    def close(self):
        """关闭连接"""
        if self._quote_ctx is not None:
            self._quote_ctx.close()
            self._quote_ctx = None
            logger.info("已关闭 Moomoo 连接")

    # ================================================================
    # 期权数据
    # ================================================================

    def get_option_expiration_dates(self, code: str = GLD_CODE) -> pd.DataFrame:
        """获取GLD所有期权到期日"""
        from moomoo import RET_OK

        ctx = self._get_quote_ctx()
        ret, data = ctx.get_option_expiration_date(code=code)

        if ret != RET_OK:
            logger.error(f"获取期权到期日失败: {data}")
            return pd.DataFrame()

        logger.info(f"{code} 期权到期日: {len(data)} 个")
        return data

    def get_option_chain(self, code: str = GLD_CODE,
                         start: str = None, end: str = None) -> pd.DataFrame:
        """
        获取GLD期权链
        返回所有可用的行权价和期权代码
        """
        from moomoo import RET_OK

        ctx = self._get_quote_ctx()

        # 如果未指定日期范围，取最近30天
        if start is None:
            start = datetime.now().strftime("%Y-%m-%d")
        if end is None:
            end = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        ret, data = ctx.get_option_chain(code=code, start=start, end=end)

        if ret != RET_OK:
            logger.error(f"获取期权链失败: {data}")
            return pd.DataFrame()

        logger.info(f"{code} 期权链: {len(data)} 个合约 ({start} ~ {end})")
        return data

    def get_option_snapshot(self, option_codes: list[str]) -> pd.DataFrame:
        """
        获取期权合约的实时快照
        包含: 价格、Greeks(Delta/Gamma/Vega/Theta/Rho)、IV、OI等

        参数:
            option_codes: 期权合约代码列表，如 ["US.GLD250321C260000"]
        """
        from moomoo import RET_OK

        ctx = self._get_quote_ctx()

        # market_snapshot 一次最多400个代码
        all_data = []
        batch_size = 400

        for i in range(0, len(option_codes), batch_size):
            batch = option_codes[i:i + batch_size]
            ret, data = ctx.get_market_snapshot(batch)

            if ret != RET_OK:
                logger.warning(f"获取快照失败 (batch {i}): {data}")
                continue

            all_data.append(data)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"期权快照: {len(result)} 个合约")
        return result

    def get_full_option_data(self, code: str = GLD_CODE,
                             num_expirations: int = 6) -> pd.DataFrame:
        """
        获取完整的期权数据：期权链 + Greeks + IV

        流程:
        1. 获取所有到期日
        2. 取最近N个到期日的期权链
        3. 获取所有合约的实时快照（含Greeks和IV）
        4. 合并返回
        """
        from moomoo import RET_OK

        # Step 1: 获取到期日
        exp_df = self.get_option_expiration_dates(code)
        if exp_df.empty:
            return pd.DataFrame()

        # 只取未来的到期日
        exp_df["strike_time"] = pd.to_datetime(exp_df["strike_time"])
        future_exp = exp_df[exp_df["option_expiry_date_distance"] >= 0]
        target_dates = future_exp.head(num_expirations)

        if target_dates.empty:
            logger.warning("没有可用的未来到期日")
            return pd.DataFrame()

        start_date = target_dates["strike_time"].iloc[0].strftime("%Y-%m-%d")
        end_date = target_dates["strike_time"].iloc[-1].strftime("%Y-%m-%d")

        logger.info(f"获取 {len(target_dates)} 个到期日的期权链: {start_date} ~ {end_date}")

        # Step 2: 获取期权链
        chain = self.get_option_chain(code, start=start_date, end=end_date)
        if chain.empty:
            return pd.DataFrame()

        # Step 3: 获取所有合约的快照
        option_codes = chain["code"].tolist()
        logger.info(f"获取 {len(option_codes)} 个合约的实时快照...")

        snapshot = self.get_option_snapshot(option_codes)
        if snapshot.empty:
            return chain  # 至少返回期权链基本信息

        # Step 4: 合并
        # 从snapshot提取期权相关字段
        option_fields = [
            "code", "name", "last_price", "open_price", "high_price", "low_price",
            "prev_close_price", "volume", "turnover",
            "option_valid", "option_type", "strike_time", "option_strike_price",
            "option_contract_size", "option_open_interest",
            "option_implied_volatility",
            "option_delta", "option_gamma", "option_vega", "option_theta", "option_rho",
            "option_premium", "option_expiry_date_distance",
        ]

        available_fields = [f for f in option_fields if f in snapshot.columns]
        result = snapshot[available_fields].copy()

        logger.info(f"完整期权数据: {len(result)} 个合约, {len(available_fields)} 个字段")
        return result

    # ================================================================
    # 行情数据
    # ================================================================

    def get_gld_history_kline(self, code: str = GLD_CODE,
                              start: str = None, end: str = None,
                              ktype: str = "K_DAY") -> pd.DataFrame:
        """
        获取GLD历史K线数据

        参数:
            ktype: K线类型 - K_DAY, K_WEEK, K_MON, K_60M, K_15M, K_5M, K_1M
        """
        from moomoo import RET_OK, KLType

        ctx = self._get_quote_ctx()

        ktype_map = {
            "K_DAY": KLType.K_DAY,
            "K_WEEK": KLType.K_WEEK,
            "K_MON": KLType.K_MON,
            "K_60M": KLType.K_60M,
            "K_15M": KLType.K_15M,
            "K_5M": KLType.K_5M,
            "K_1M": KLType.K_1M,
        }

        kl_type = ktype_map.get(ktype, KLType.K_DAY)

        if start is None:
            start = "2004-11-18"
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        all_data = []
        page_req_key = None

        while True:
            ret, data, page_req_key = ctx.request_history_kline(
                code=code,
                start=start,
                end=end,
                ktype=kl_type,
                max_count=1000,
                page_req_key=page_req_key,
            )

            if ret != RET_OK:
                logger.error(f"获取K线失败: {data}")
                break

            all_data.append(data)

            if page_req_key is None:
                break

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"{code} {ktype} K线: {len(result)} 条")
        return result

    def get_realtime_quote(self, codes: list[str] = None) -> pd.DataFrame:
        """获取实时报价"""
        from moomoo import RET_OK, SubType

        if codes is None:
            codes = [GLD_CODE]

        ctx = self._get_quote_ctx()

        # 先订阅
        ret, err = ctx.subscribe(codes, [SubType.QUOTE])
        if ret != RET_OK:
            logger.error(f"订阅失败: {err}")
            return pd.DataFrame()

        # 获取报价
        ret, data = ctx.get_stock_quote(codes)
        if ret != RET_OK:
            logger.error(f"获取报价失败: {data}")
            return pd.DataFrame()

        return data

    # ================================================================
    # 保存
    # ================================================================

    def save_option_data(self, data: pd.DataFrame, label: str = "") -> str:
        """保存期权数据快照"""
        moomoo_dir = os.path.join(self.raw_data_path, "moomoo_options")
        os.makedirs(moomoo_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gld_options_{label}_{timestamp}.csv" if label else \
                   f"gld_options_{timestamp}.csv"
        filepath = os.path.join(moomoo_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"已保存期权数据 -> {filepath}")
        return filepath


def run_moomoo_data_collection():
    """执行 Moomoo 数据采集"""
    print("=" * 60)
    print("采集 Moomoo API 数据")
    print("=" * 60)

    collector = MoomooDataCollector()

    try:
        # 1. 获取期权到期日
        print("\n[1] 获取 GLD 期权到期日...")
        exp_dates = collector.get_option_expiration_dates()
        if not exp_dates.empty:
            print(f"  到期日: {len(exp_dates)} 个")
            print(exp_dates.head(8).to_string())

        # 2. 获取完整期权数据（含Greeks和IV）
        print("\n[2] 获取 GLD 完整期权数据 (含Greeks/IV)...")
        option_data = collector.get_full_option_data(num_expirations=6)
        if not option_data.empty:
            print(f"  合约数: {len(option_data)}")
            print(f"  字段: {list(option_data.columns)}")
            collector.save_option_data(option_data, label="full")

            # 显示IV统计
            if "option_implied_volatility" in option_data.columns:
                iv = option_data["option_implied_volatility"].dropna()
                print(f"\n  IV统计: mean={iv.mean():.2f}%, "
                      f"min={iv.min():.2f}%, max={iv.max():.2f}%")

        # 3. GLD 实时报价
        print("\n[3] 获取 GLD 实时报价...")
        quote = collector.get_realtime_quote()
        if not quote.empty:
            print(quote.to_string())

    except ConnectionRefusedError:
        print("\n错误: 无法连接到 Moomoo OpenD")
        print("请确保:")
        print("  1. 已安装并启动 Moomoo OpenD 客户端")
        print("  2. OpenD 监听端口为 11111 (默认)")
        print("  3. 已在 Moomoo 客户端登录账号")
    except Exception as e:
        print(f"\n错误: {e}")
        logger.exception("Moomoo 数据采集异常")
    finally:
        collector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_moomoo_data_collection()
