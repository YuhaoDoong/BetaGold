"""
GLD 期权历史价格数据库构建

从 Moomoo API 批量获取所有当前上市的 GLD 期权合约的历史日K线,
构建期权价格数据库用于 Phase 4B 回测。

Moomoo API 限制:
- 只能获取当前上市的合约 (已到期的合约无法获取)
- 历史K线最多约1年 (取决于合约上市时间和交易活跃度)
- 频率限制: 每30秒最多30次请求

输出:
- data/raw/options_history/kline_db/all_klines.parquet
  columns: code, date, open, high, low, close, volume, strike, expiry, option_type, dte_at_date

使用:
    conda run -n gold python -m src.data.options.options_history_builder
"""

import os
import sys
import time
import logging
import re
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_option_code(code: str) -> dict:
    """
    解析 Moomoo 期权代码, 提取 strike/expiry/type.

    例: US.GLD260618C276000
    → expiry=2026-06-18, type=CALL, strike=276.0
    """
    m = re.match(
        r'US\.GLD(\d{2})(\d{2})(\d{2})([CP])(\d+)', code)
    if not m:
        return {}
    yy, mm, dd, cp, strike_raw = m.groups()
    expiry = f"20{yy}-{mm}-{dd}"
    option_type = "CALL" if cp == "C" else "PUT"
    strike = int(strike_raw) / 1000.0
    return {
        "expiry": expiry,
        "option_type": option_type,
        "strike": strike,
    }


class OptionsHistoryBuilder:
    """批量构建 GLD 期权历史价格数据库"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        self.db_dir = os.path.join(
            self.raw_data_path, "options_history", "kline_db")
        os.makedirs(self.db_dir, exist_ok=True)

    def _get_all_option_codes(self, min_dte: int = 0,
                              strike_low: float = 0,
                              strike_high: float = 9999) -> pd.DataFrame:
        """获取当前上市的 GLD 期权合约代码 (可筛选)"""
        from src.data.options.moomoo_data import MoomooDataCollector
        collector = MoomooDataCollector(config=self.config)

        try:
            exp_df = collector.get_option_expiration_dates()
            if exp_df.empty:
                return pd.DataFrame()

            exp_df["strike_time"] = pd.to_datetime(exp_df["strike_time"])
            # 筛选 DTE
            future = exp_df[
                exp_df["option_expiry_date_distance"] >= min_dte]

            all_chains = []
            batch_dates = sorted(future["strike_time"].tolist())

            # 按30天窗口分批获取期权链
            i = 0
            while i < len(batch_dates):
                batch_start = batch_dates[i]
                batch_end = batch_start + pd.Timedelta(days=29)
                batch = [d for d in batch_dates[i:] if d <= batch_end]

                if not batch:
                    i += 1
                    continue

                start_str = batch[0].strftime("%Y-%m-%d")
                end_str = batch[-1].strftime("%Y-%m-%d")
                logger.info(f"  获取期权链: {start_str} ~ {end_str}"
                            f" ({len(batch)} 个到期日)")

                chain = collector.get_option_chain(start=start_str, end=end_str)
                if not chain.empty:
                    all_chains.append(chain)

                i += len(batch)
                if i < len(batch_dates):
                    time.sleep(3.5)

            if not all_chains:
                return pd.DataFrame()

            result = pd.concat(all_chains, ignore_index=True)
            result = result.drop_duplicates(subset=["code"])

            # 按 strike 范围筛选
            if strike_low > 0 or strike_high < 9999:
                result["_strike"] = result["code"].apply(
                    lambda c: parse_option_code(c).get("strike", 0))
                before = len(result)
                result = result[
                    (result["_strike"] >= strike_low) &
                    (result["_strike"] <= strike_high)
                ].drop(columns=["_strike"])
                logger.info(f"Strike ${strike_low}-${strike_high} "
                            f"筛选: {before} → {len(result)}")

            logger.info(f"总计 {len(result)} 个期权合约")
            return result

        finally:
            collector.close()

    def _fetch_klines_batch(self, codes: list[str],
                            start: str = "2024-06-01") -> pd.DataFrame:
        """
        批量获取期权合约的历史日K线.

        频率控制: 每30秒最多30次 → 每次间隔约1秒
        """
        from moomoo import OpenQuoteContext, RET_OK

        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        end = datetime.now().strftime("%Y-%m-%d")

        all_klines = []
        total = len(codes)
        success = 0
        empty = 0
        fail = 0

        try:
            for idx, code in enumerate(codes):
                if (idx + 1) % 100 == 0 or idx == 0:
                    logger.info(f"  进度: {idx+1}/{total} "
                                f"(成功={success}, 空={empty}, 失败={fail})")

                try:
                    ret, data, _ = ctx.request_history_kline(
                        code, start=start, end=end,
                        ktype="K_DAY", max_count=1000)

                    if ret != RET_OK:
                        fail += 1
                        continue

                    if data.empty:
                        empty += 1
                        continue

                    # 解析期权代码
                    parsed = parse_option_code(code)
                    if not parsed:
                        continue

                    df = data[["time_key", "open", "high", "low",
                               "close", "volume"]].copy()
                    df.rename(columns={"time_key": "date"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df["code"] = code
                    df["strike"] = parsed["strike"]
                    df["expiry"] = parsed["expiry"]
                    df["option_type"] = parsed["option_type"]

                    # 计算每个日期的DTE
                    expiry_date = pd.Timestamp(parsed["expiry"]).date()
                    df["dte_at_date"] = df["date"].apply(
                        lambda d: (expiry_date - d).days)

                    all_klines.append(df)
                    success += 1

                except Exception as e:
                    fail += 1
                    if fail <= 5:
                        logger.warning(f"  获取 {code} 失败: {e}")

                # 频率控制: ~1秒/请求
                time.sleep(1.05)

        finally:
            ctx.close()

        logger.info(f"  完成: 成功={success}, 空={empty}, 失败={fail}")

        if not all_klines:
            return pd.DataFrame()

        return pd.concat(all_klines, ignore_index=True)

    def build(self, start: str = "2024-06-01",
              max_contracts: int = None,
              min_dte: int = 30,
              strike_low: float = 200,
              strike_high: float = 320) -> str:
        """
        构建完整的期权历史价格数据库.

        Args:
            start: 历史起始日期
            max_contracts: 最大合约数 (None=全部, 用于测试)
            min_dte: 最小DTE筛选 (跳过近期周合约)
            strike_low/high: 行权价范围
        Returns:
            保存路径
        """
        logger.info("=" * 60)
        logger.info("构建 GLD 期权历史价格数据库")
        logger.info("=" * 60)

        # Step 1: 获取合约代码 (带筛选)
        logger.info(f"[1] 获取期权合约 (DTE>={min_dte}, "
                    f"strike ${strike_low}-${strike_high})...")
        chain = self._get_all_option_codes(
            min_dte=min_dte,
            strike_low=strike_low,
            strike_high=strike_high)
        if chain.empty:
            logger.error("获取合约列表失败")
            return ""

        codes = chain["code"].tolist()
        if max_contracts:
            codes = codes[:max_contracts]
        logger.info(f"  将获取 {len(codes)} 个合约的历史K线")

        # 估算时间
        est_minutes = len(codes) * 1.1 / 60
        logger.info(f"  预计耗时: ~{est_minutes:.0f} 分钟")

        # Step 2: 批量获取历史K线
        logger.info(f"[2] 批量获取历史K线 (从 {start})...")
        kline_df = self._fetch_klines_batch(codes, start=start)

        if kline_df.empty:
            logger.error("获取K线数据失败")
            return ""

        # Step 3: 保存
        logger.info("[3] 保存数据库...")
        output_path = os.path.join(self.db_dir, "all_klines.parquet")
        kline_df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"  已保存: {output_path}")

        # CSV备份 (可选, 方便查看)
        csv_path = os.path.join(self.db_dir, "all_klines.csv")
        kline_df.to_csv(csv_path, index=False)

        # 统计
        self._print_summary(kline_df)

        return output_path

    def build_incremental(self, start: str = "2024-06-01") -> str:
        """
        增量更新: 只获取数据库中没有的合约.
        """
        existing_path = os.path.join(self.db_dir, "all_klines.parquet")
        existing_codes = set()

        if os.path.exists(existing_path):
            existing = pd.read_parquet(existing_path)
            existing_codes = set(existing["code"].unique())
            logger.info(f"已有数据库: {len(existing_codes)} 个合约")
        else:
            existing = pd.DataFrame()

        # 获取所有当前合约
        chain = self._get_all_option_codes()
        if chain.empty:
            return ""

        all_codes = chain["code"].tolist()
        new_codes = [c for c in all_codes if c not in existing_codes]

        if not new_codes:
            logger.info("所有合约已在数据库中，无需更新")
            # 但仍然需要更新最新几天的数据
            # 获取最新日期
            if not existing.empty:
                last_date = existing["date"].max()
                logger.info(f"更新从 {last_date} 之后的数据...")
                update_df = self._fetch_klines_batch(
                    all_codes[:200], start=str(last_date))
                if not update_df.empty:
                    combined = pd.concat([existing, update_df],
                                         ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["code", "date"], keep="last")
                    combined.to_parquet(existing_path, index=False,
                                       engine="pyarrow")
                    logger.info(f"更新完成: {len(combined)} 行")
            return existing_path

        logger.info(f"新增 {len(new_codes)} 个合约待获取")

        # 获取新合约的K线
        new_klines = self._fetch_klines_batch(new_codes, start=start)

        if not new_klines.empty:
            combined = pd.concat([existing, new_klines], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["code", "date"], keep="last")
            combined.to_parquet(existing_path, index=False, engine="pyarrow")
            logger.info(f"增量更新完成: {len(combined)} 行")
            self._print_summary(combined)
            return existing_path

        return existing_path if not existing.empty else ""

    def _print_summary(self, df: pd.DataFrame):
        """打印数据库统计"""
        print(f"\n{'=' * 60}")
        print("  GLD 期权历史价格数据库")
        print(f"{'=' * 60}")
        print(f"  总行数: {len(df):,}")
        print(f"  合约数: {df['code'].nunique():,}")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")

        calls = df[df["option_type"] == "CALL"]
        puts = df[df["option_type"] == "PUT"]
        print(f"  Call: {calls['code'].nunique()} 合约, "
              f"{len(calls):,} 行")
        print(f"  Put:  {puts['code'].nunique()} 合约, "
              f"{len(puts):,} 行")

        expiries = sorted(df["expiry"].unique())
        print(f"  到期日: {len(expiries)} 个")
        print(f"    最近: {expiries[0]}")
        print(f"    最远: {expiries[-1]}")

        # 每月数据量
        df_copy = df.copy()
        df_copy["month"] = pd.to_datetime(df_copy["date"]).dt.to_period("M")
        monthly = df_copy.groupby("month").agg(
            days=("date", "nunique"),
            contracts=("code", "nunique"),
            rows=("code", "count"),
        )
        print("\n  月度数据量:")
        for idx, row in monthly.iterrows():
            print(f"    {idx}: {int(row['days']):2d} 天, "
                  f"{int(row['contracts']):4d} 合约, "
                  f"{int(row['rows']):6d} 行")

        size_mb = os.path.getsize(
            os.path.join(self.db_dir, "all_klines.parquet")) / 1024 / 1024
        print(f"\n  文件大小: {size_mb:.1f} MB")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="构建 GLD 期权历史价格数据库")
    parser.add_argument("--start", default="2024-06-01",
                        help="历史起始日期 (default: 2024-06-01)")
    parser.add_argument("--max", type=int, default=None,
                        help="最大合约数 (用于测试)")
    parser.add_argument("--min-dte", type=int, default=30,
                        help="最小DTE筛选 (default: 30)")
    parser.add_argument("--strike-low", type=float, default=200,
                        help="最低strike (default: 200)")
    parser.add_argument("--strike-high", type=float, default=320,
                        help="最高strike (default: 320)")
    parser.add_argument("--incremental", action="store_true",
                        help="增量更新模式")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    builder = OptionsHistoryBuilder()

    if args.incremental:
        path = builder.build_incremental(start=args.start)
    else:
        path = builder.build(
            start=args.start, max_contracts=args.max,
            min_dte=args.min_dte,
            strike_low=args.strike_low,
            strike_high=args.strike_high)

    if path:
        print(f"\n数据库路径: {path}")
    else:
        print("\n构建失败")


if __name__ == "__main__":
    main()
