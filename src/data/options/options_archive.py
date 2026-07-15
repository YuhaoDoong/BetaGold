"""
历史期权链存档系统 (Phase 1.5)

每日自动存档 GLD 期权链完整快照，用于支撑期权策略回测。
没有历史期权链，回测只能用 Black-Scholes 反推模拟价格，不可靠。

存档内容:
- 日终全链快照: 所有到期日 × 所有 strike × bid/ask/mid/IV/Greeks/OI/volume/DTE
- 盘中关键横截面: 最近3个月到期 × 关键 delta (0.25/0.50/0.75) 的 call/put

存储格式: Parquet (压缩效率高)
存储路径: data/raw/options_history/YYYY-MM-DD/

使用:
    # 日终全链快照
    python -m src.data.options_archive

    # 盘中横截面
    python -m src.data.options_archive --intraday

    # 定时任务 (crontab):
    # 每个交易日 16:05 ET 存日终快照
    # 5 16 * * 1-5 cd /Users/yhdong/Gold && conda run -n gold python -m src.data.options_archive
    # 每个交易日盘中每2小时存横截面
    # 30 11,13,15 * * 1-5 cd /Users/yhdong/Gold && conda run -n gold python -m src.data.options_archive --intraday
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config

logger = logging.getLogger(__name__)

# Moomoo snapshot 中需要提取的字段
SNAPSHOT_FIELDS = [
    "code", "name", "update_time",
    "last_price", "open_price", "high_price", "low_price",
    "prev_close_price", "volume", "turnover",
    "bid_price", "ask_price", "bid_vol", "ask_vol",
    "option_valid", "option_type", "strike_time", "option_strike_price",
    "option_contract_size", "option_open_interest",
    "option_implied_volatility",
    "option_delta", "option_gamma", "option_vega", "option_theta", "option_rho",
    "option_premium", "option_expiry_date_distance",
    "option_net_open_interest",
]


class OptionsArchiver:
    """历史期权链存档器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_data_path = self.config["paths"]["raw_data"]
        self.archive_dir = os.path.join(self.raw_data_path, "options_history")
        os.makedirs(self.archive_dir, exist_ok=True)

    def _get_moomoo_collector(self):
        """获取 Moomoo 采集器"""
        from src.data.options.moomoo_data import MoomooDataCollector
        return MoomooDataCollector(config=self.config)

    def _get_full_snapshot(self, collector, num_expirations: int = 30) -> pd.DataFrame:
        """
        获取完整期权链快照（包含 bid/ask）

        与 MoomooDataCollector.get_full_option_data 类似，
        但提取更多字段（bid_price, ask_price 等）。
        Moomoo get_option_chain API 限制每次请求不超过 30 天，
        因此按到期日分批请求。
        """
        import time

        # Step 1: 获取到期日
        exp_df = collector.get_option_expiration_dates()
        if exp_df.empty:
            return pd.DataFrame()

        exp_df["strike_time"] = pd.to_datetime(exp_df["strike_time"])
        future_exp = exp_df[exp_df["option_expiry_date_distance"] >= 0]
        target_dates = future_exp.head(num_expirations)

        if target_dates.empty:
            logger.warning("没有可用的未来到期日")
            return pd.DataFrame()

        all_exp_dates = sorted(target_dates["strike_time"].tolist())
        logger.info(f"获取 {len(all_exp_dates)} 个到期日: "
                    f"{all_exp_dates[0].strftime('%Y-%m-%d')} ~ "
                    f"{all_exp_dates[-1].strftime('%Y-%m-%d')}")

        # Step 2: 按 30 天窗口分批获取期权链
        all_chains = []
        batch_start = all_exp_dates[0]
        i = 0

        while i < len(all_exp_dates):
            # 找出从 batch_start 起 30 天内的所有到期日
            batch_end = batch_start + pd.Timedelta(days=29)
            batch_dates = [d for d in all_exp_dates[i:] if d <= batch_end]

            if not batch_dates:
                # 下一个日期超出 30 天窗口，以它为新起点
                batch_start = all_exp_dates[i]
                continue

            start_str = batch_dates[0].strftime("%Y-%m-%d")
            end_str = batch_dates[-1].strftime("%Y-%m-%d")
            logger.info(f"  批次: {start_str} ~ {end_str} ({len(batch_dates)} 个到期日)")

            chain = collector.get_option_chain(start=start_str, end=end_str)
            if not chain.empty:
                all_chains.append(chain)

            i += len(batch_dates)
            if i < len(all_exp_dates):
                batch_start = all_exp_dates[i]
                time.sleep(3.5)  # Moomoo 限制每30秒最多10次

        if not all_chains:
            logger.error("所有批次期权链获取失败")
            return pd.DataFrame()

        full_chain = pd.concat(all_chains, ignore_index=True)
        full_chain = full_chain.drop_duplicates(subset=["code"])
        logger.info(f"期权链合计: {len(full_chain)} 个合约")

        # Step 3: 获取所有合约的快照（含 bid/ask），分批 400 个
        option_codes = full_chain["code"].tolist()
        logger.info(f"获取 {len(option_codes)} 个合约的快照...")

        snapshot = collector.get_option_snapshot(option_codes)
        if snapshot.empty:
            return pd.DataFrame()

        # Step 4: 提取可用字段
        available = [f for f in SNAPSHOT_FIELDS if f in snapshot.columns]
        result = snapshot[available].copy()

        # Step 5: 添加标准化衍生字段
        result = self._standardize(result)

        logger.info(f"完整快照: {len(result)} 个合约, {len(result.columns)} 个字段")
        return result

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加标准化衍生字段:
        - price_mid: (bid + ask) / 2
        - spread_abs: ask - bid
        - spread_pct: spread / mid * 100
        - iv_decimal: IV 从百分比转小数
        - dte: 到期剩余天数（整数）
        - snapshot_time: 快照时间戳 (UTC)
        """
        # mid price
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["price_mid"] = (df["bid_price"] + df["ask_price"]) / 2
            df["spread_abs"] = df["ask_price"] - df["bid_price"]
            # 避免除零
            mid_nonzero = df["price_mid"].replace(0, float("nan"))
            df["spread_pct"] = (df["spread_abs"] / mid_nonzero * 100).round(4)
        else:
            # 没有 bid/ask 时用 last_price 作为近似
            df["price_mid"] = df.get("last_price", 0)
            df["spread_abs"] = float("nan")
            df["spread_pct"] = float("nan")

        # IV: 百分比 -> 小数
        if "option_implied_volatility" in df.columns:
            df["iv_decimal"] = df["option_implied_volatility"] / 100.0

        # DTE
        if "option_expiry_date_distance" in df.columns:
            df["dte"] = df["option_expiry_date_distance"].astype(int)

        # 快照时间 (UTC)
        df["snapshot_time_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        return df

    def save_eod_snapshot(self) -> str:
        """
        保存日终全链快照

        保存路径: options_history/YYYY-MM-DD/eod_full.parquet
        同时保存 CSV 备份（方便人工查看）
        """
        collector = self._get_moomoo_collector()

        try:
            logger.info("=== 日终全链快照开始 ===")
            df = self._get_full_snapshot(collector, num_expirations=30)

            if df.empty:
                logger.error("获取快照失败，无数据")
                return ""

            # 按日期创建目录
            today = datetime.now().strftime("%Y-%m-%d")
            day_dir = os.path.join(self.archive_dir, today)
            os.makedirs(day_dir, exist_ok=True)

            # 保存 Parquet
            parquet_path = os.path.join(day_dir, "eod_full.parquet")
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            logger.info(f"已保存 Parquet -> {parquet_path}")

            # 保存 CSV 备份
            csv_path = os.path.join(day_dir, "eod_full.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"已保存 CSV -> {csv_path}")

            # 打印摘要统计
            self._print_summary(df, "日终全链快照")

            return parquet_path

        except ConnectionRefusedError:
            logger.error("Moomoo OpenD 未运行，无法获取期权数据")
            print("\n请先启动 OpenD:")
            print("  nohup /Users/yhdong/Software/moomoo_OpenD_9.6.5618_Mac/"
                  "moomoo_OpenD_9.6.5618_Mac/OpenD.app/Contents/MacOS/OpenD &")
            return ""
        except Exception as e:
            logger.exception(f"日终快照失败: {e}")
            return ""
        finally:
            collector.close()

    def save_intraday_crosssection(self) -> str:
        """
        保存盘中关键横截面

        筛选: 最近3个月到期 × 关键 delta (0.25/0.50/0.75) 的 call/put
        保存路径: options_history/YYYY-MM-DD/intraday_HHMMSS.parquet
        """
        collector = self._get_moomoo_collector()

        try:
            logger.info("=== 盘中横截面快照开始 ===")
            # 只取最近6个到期日（约3个月）
            df = self._get_full_snapshot(collector, num_expirations=6)

            if df.empty:
                logger.error("获取快照失败，无数据")
                return ""

            # 筛选关键 delta 合约
            key_deltas = [0.25, 0.50, 0.75]
            delta_tolerance = 0.08  # 容差

            if "option_delta" not in df.columns:
                logger.warning("无 delta 数据，保存全部合约")
                filtered = df
            else:
                masks = []
                for target_d in key_deltas:
                    # Call: delta 接近 target
                    call_mask = (
                        (df["option_type"] == "CALL") &
                        (df["option_delta"].between(target_d - delta_tolerance,
                                                     target_d + delta_tolerance))
                    )
                    # Put: delta 接近 -target
                    put_mask = (
                        (df["option_type"] == "PUT") &
                        (df["option_delta"].between(-target_d - delta_tolerance,
                                                     -target_d + delta_tolerance))
                    )
                    masks.append(call_mask | put_mask)

                combined_mask = masks[0]
                for m in masks[1:]:
                    combined_mask = combined_mask | m

                filtered = df[combined_mask].copy()

                if filtered.empty:
                    logger.warning("关键 delta 筛选后无合约，保存全部合约")
                    filtered = df

            # 按日期创建目录
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%H%M%S")
            day_dir = os.path.join(self.archive_dir, today)
            os.makedirs(day_dir, exist_ok=True)

            # 保存
            parquet_path = os.path.join(day_dir, f"intraday_{timestamp}.parquet")
            filtered.to_parquet(parquet_path, index=False, engine="pyarrow")
            logger.info(f"已保存横截面 -> {parquet_path}")

            csv_path = os.path.join(day_dir, f"intraday_{timestamp}.csv")
            filtered.to_csv(csv_path, index=False)

            self._print_summary(filtered, f"盘中横截面 ({timestamp})")

            return parquet_path

        except ConnectionRefusedError:
            logger.error("Moomoo OpenD 未运行")
            return ""
        except Exception as e:
            logger.exception(f"盘中横截面快照失败: {e}")
            return ""
        finally:
            collector.close()

    def _print_summary(self, df: pd.DataFrame, label: str):
        """打印快照摘要统计"""
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")
        print(f"  合约数: {len(df)}")

        if "option_type" in df.columns:
            calls = len(df[df["option_type"] == "CALL"])
            puts = len(df[df["option_type"] == "PUT"])
            print(f"  Call: {calls} | Put: {puts}")

        if "strike_time" in df.columns:
            expirations = df["strike_time"].nunique()
            print(f"  到期日: {expirations} 个")

        if "iv_decimal" in df.columns:
            iv = df["iv_decimal"].dropna()
            iv_valid = iv[iv > 0]
            if not iv_valid.empty:
                print(f"  IV: mean={iv_valid.mean():.4f}, "
                      f"min={iv_valid.min():.4f}, max={iv_valid.max():.4f}")

        if "spread_pct" in df.columns:
            sp = df["spread_pct"].dropna()
            sp_valid = sp[sp >= 0]
            if not sp_valid.empty:
                print(f"  Spread%: median={sp_valid.median():.2f}%, "
                      f"mean={sp_valid.mean():.2f}%")

        if "option_open_interest" in df.columns:
            oi = df["option_open_interest"]
            print(f"  OI: total={int(oi.sum())}, "
                  f"median={int(oi.median())}, max={int(oi.max())}")

        if "dte" in df.columns:
            print(f"  DTE: {int(df['dte'].min())} ~ {int(df['dte'].max())} 天")

        # 存档大小
        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  内存占用: {size_mb:.2f} MB")

    def list_archive(self) -> pd.DataFrame:
        """列出所有已存档的日期和文件"""
        records = []
        if not os.path.exists(self.archive_dir):
            return pd.DataFrame()

        for date_dir in sorted(os.listdir(self.archive_dir)):
            day_path = os.path.join(self.archive_dir, date_dir)
            if not os.path.isdir(day_path):
                continue
            for fname in sorted(os.listdir(day_path)):
                fpath = os.path.join(day_path, fname)
                size_kb = os.path.getsize(fpath) / 1024
                records.append({
                    "date": date_dir,
                    "file": fname,
                    "size_kb": round(size_kb, 1),
                })

        return pd.DataFrame(records)

    def load_eod_snapshot(self, date: str) -> pd.DataFrame:
        """加载指定日期的日终快照"""
        parquet_path = os.path.join(self.archive_dir, date, "eod_full.parquet")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)

        csv_path = os.path.join(self.archive_dir, date, "eod_full.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)

        logger.warning(f"未找到 {date} 的日终快照")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="GLD 期权链历史存档")
    parser.add_argument("--intraday", action="store_true",
                        help="保存盘中关键横截面（默认保存日终全链快照）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有已存档日期")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    archiver = OptionsArchiver()

    if args.list:
        archive = archiver.list_archive()
        if archive.empty:
            print("尚无存档数据")
        else:
            print(archive.to_string(index=False))
        return

    if args.intraday:
        path = archiver.save_intraday_crosssection()
    else:
        path = archiver.save_eod_snapshot()

    if path:
        print(f"\n存档完成: {path}")
    else:
        print("\n存档失败")


if __name__ == "__main__":
    main()
