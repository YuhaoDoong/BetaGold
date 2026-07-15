"""
批量下载 GLD 期权历史K线 — 优化版

只下载关键到期日 + ATM附近的合约, 减少请求次数.
"""
import os
import sys
import re
import time
import logging

import pandas as pd
from moomoo import OpenQuoteContext, RET_OK

PROJECT_ROOT = "/Users/yhdong/Gold"
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_option_code(code):
    m = re.match(r'US\.GLD(\d{2})(\d{2})(\d{2})([CP])(\d+)', code)
    if not m:
        return {}
    yy, mm, dd, cp, strike_raw = m.groups()
    return {
        "expiry": f"20{yy}-{mm}-{dd}",
        "option_type": "CALL" if cp == "C" else "PUT",
        "strike": int(strike_raw) / 1000.0,
    }


def main():
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

    try:
        # 1. 获取所有到期日
        ret, exp_df = ctx.get_option_expiration_date("US.GLD")
        if ret != RET_OK:
            logger.error(f"获取到期日失败: {exp_df}")
            return

        exp_df["strike_time"] = pd.to_datetime(exp_df["strike_time"])
        future = exp_df[exp_df["option_expiry_date_distance"] >= 14]
        all_exp_dates = sorted(future["strike_time"].tolist())
        logger.info(f"到期日: {len(all_exp_dates)} 个")

        # 2. 获取所有期权链
        all_chains = []
        i = 0
        while i < len(all_exp_dates):
            batch_start = all_exp_dates[i]
            batch_end = batch_start + pd.Timedelta(days=29)
            batch = [d for d in all_exp_dates[i:] if d <= batch_end]
            if not batch:
                i += 1
                continue
            start_str = batch[0].strftime("%Y-%m-%d")
            end_str = batch[-1].strftime("%Y-%m-%d")
            logger.info(f"  期权链: {start_str} ~ {end_str} ({len(batch)} 个)")
            ret2, chain = ctx.get_option_chain(
                "US.GLD", start=start_str, end=end_str)
            if ret2 == RET_OK and not chain.empty:
                all_chains.append(chain)
            i += len(batch)
            if i < len(all_exp_dates):
                time.sleep(3.5)

        full = pd.concat(all_chains, ignore_index=True)
        full = full.drop_duplicates(subset=["code"])
        logger.info(f"总合约: {len(full)}")

        # 3. 筛选: strike $230-$550
        full["_strike"] = full["code"].apply(
            lambda c: parse_option_code(c).get("strike", 0))
        filtered = full[
            (full["_strike"] >= 230) & (full["_strike"] <= 550)
        ].copy()
        logger.info(f"Strike $230-$550: {len(filtered)} 合约")

        # 4. 批量获取K线
        codes = filtered["code"].tolist()
        total = len(codes)
        logger.info(f"开始下载 {total} 个合约的历史K线...")
        logger.info(f"预计耗时: ~{total * 1.1 / 60:.0f} 分钟")

        all_klines = []
        success = 0
        empty = 0
        fail = 0

        for idx, code in enumerate(codes):
            if (idx + 1) % 200 == 0 or idx == 0:
                logger.info(f"  进度: {idx+1}/{total} "
                            f"(ok={success}, empty={empty}, fail={fail})")

            try:
                ret3, data, _ = ctx.request_history_kline(
                    code, start="2024-06-01",
                    end="2026-03-11", ktype="K_DAY", max_count=1000)

                if ret3 != RET_OK:
                    fail += 1
                    if fail <= 10:
                        logger.warning(f"  {code} 返回错误: {data}")
                    time.sleep(1.05)
                    continue

                if data.empty:
                    empty += 1
                    time.sleep(1.05)
                    continue

                parsed = parse_option_code(code)
                if not parsed:
                    time.sleep(1.05)
                    continue

                df = data[["time_key", "open", "high", "low",
                           "close", "volume"]].copy()
                df.rename(columns={"time_key": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["code"] = code
                df["strike"] = parsed["strike"]
                df["expiry"] = parsed["expiry"]
                df["option_type"] = parsed["option_type"]

                expiry_date = pd.Timestamp(parsed["expiry"]).date()
                df["dte_at_date"] = df["date"].apply(
                    lambda d: (expiry_date - d).days)

                all_klines.append(df)
                success += 1

            except Exception as e:
                fail += 1
                if fail <= 10:
                    logger.warning(f"  {code} 异常: {e}")

            time.sleep(1.05)

        logger.info(f"下载完成: ok={success}, empty={empty}, fail={fail}")

        if not all_klines:
            logger.error("无数据")
            return

        result = pd.concat(all_klines, ignore_index=True)

        # 5. 保存
        db_dir = os.path.join(
            PROJECT_ROOT, "data", "raw", "options_history", "kline_db")
        os.makedirs(db_dir, exist_ok=True)

        parquet_path = os.path.join(db_dir, "all_klines.parquet")
        result.to_parquet(parquet_path, index=False, engine="pyarrow")
        csv_path = os.path.join(db_dir, "all_klines.csv")
        result.to_csv(csv_path, index=False)

        logger.info(f"已保存: {parquet_path}")
        logger.info(f"  总行数: {len(result):,}")
        logger.info(f"  合约数: {result['code'].nunique()}")
        logger.info(f"  日期: {result['date'].min()} ~ {result['date'].max()}")
        logger.info(f"  大小: {os.path.getsize(parquet_path)/1024/1024:.1f} MB")

        # 月度统计
        result_copy = result.copy()
        result_copy["month"] = pd.to_datetime(
            result_copy["date"]).dt.to_period("M")
        for m, g in result_copy.groupby("month"):
            logger.info(f"  {m}: {g['date'].nunique():2d} 天, "
                        f"{g['code'].nunique():4d} 合约, "
                        f"{len(g):6d} 行")

    finally:
        ctx.close()


if __name__ == "__main__":
    main()
