"""
智能K线下载 — 额度管理版

每天检查可用额度, 优先下载ATM附近的高价值合约.
配合 crontab 每天运行, 逐步积累历史K线数据库.

优先级:
1. 季度到期 (Jun/Sep/Dec) — 历史最长
2. ATM附近 strikes ($380-$520 根据当前GLD价格)
3. Call + Put 成对下载

使用:
    conda run -n gold python scripts/smart_kline_download.py

Crontab (每天5:30am, 在EOD快照之后):
    30 5 * * 2-6 cd /Users/yhdong/Gold && conda run -n gold python scripts/smart_kline_download.py >> logs/smart_kline.log 2>&1
"""
import os
import sys
import re
import time
import logging
from datetime import datetime

import pandas as pd
from moomoo import OpenQuoteContext, RET_OK

PROJECT_ROOT = "/Users/yhdong/Gold"
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "options_history", "kline_db",
    "all_klines.parquet")


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


def get_priority_contracts(ctx):
    """获取按优先级排序的待下载合约列表"""
    # 获取已下载的合约
    existing_codes = set()
    if os.path.exists(DB_PATH):
        existing = pd.read_parquet(DB_PATH)
        existing_codes = set(existing["code"].unique())
        logger.info(f"已有数据库: {len(existing_codes)} 合约")

    # 获取当前GLD价格 (用于确定ATM范围)
    gld = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", "market", "gld.csv"),
        index_col=0, parse_dates=True)
    current_price = gld["Close"].iloc[-1]
    logger.info(f"GLD 最新价格: ${current_price:.1f}")

    # ATM范围: 当前价格 ±15%
    strike_low = current_price * 0.85
    strike_high = current_price * 1.15
    logger.info(f"ATM 目标范围: ${strike_low:.0f}-${strike_high:.0f}")

    # 获取期权链
    ret, exp_df = ctx.get_option_expiration_date("US.GLD")
    if ret != RET_OK:
        return []

    exp_df["strike_time"] = pd.to_datetime(exp_df["strike_time"])
    future = exp_df[exp_df["option_expiry_date_distance"] >= 14]

    # 按优先级排序到期日: 季度 > 月度 > 周度
    # 季度到期 (Mar/Jun/Sep/Dec) 有更长历史
    quarterly_months = {3, 6, 9, 12}
    sorted_exp = []
    for _, row in future.iterrows():
        dt = row["strike_time"]
        month = dt.month
        dte = int(row["option_expiry_date_distance"])
        # 优先级分数: 季度高, DTE 60-365 最佳
        priority = 0
        if month in quarterly_months:
            priority += 100
        if 60 <= dte <= 365:
            priority += 50  # 历史数据最多的范围
        elif 30 <= dte < 60:
            priority += 30
        sorted_exp.append((dt, dte, priority))

    sorted_exp.sort(key=lambda x: -x[2])

    # 获取每个到期日的期权链, 筛选ATM strikes
    all_codes = []
    batch_dates = [x[0] for x in sorted_exp]

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
        ret2, chain = ctx.get_option_chain(
            "US.GLD", start=start_str, end=end_str)

        if ret2 == RET_OK and not chain.empty:
            for _, row in chain.iterrows():
                code = row["code"]
                if code in existing_codes:
                    continue  # 已下载
                parsed = parse_option_code(code)
                if not parsed:
                    continue
                strike = parsed["strike"]
                if strike_low <= strike <= strike_high:
                    exp = parsed["expiry"]
                    exp_dt = pd.Timestamp(exp)
                    month = exp_dt.month
                    dte = (exp_dt - pd.Timestamp.now()).days
                    priority = 0
                    if month in quarterly_months:
                        priority += 100
                    if 60 <= dte <= 365:
                        priority += 50
                    # 越接近ATM优先级越高
                    atm_dist = abs(strike - current_price)
                    priority -= atm_dist * 0.1
                    all_codes.append((code, priority, strike, exp))

        i += len(batch)
        if i < len(batch_dates):
            time.sleep(3.5)

    # 按优先级排序
    all_codes.sort(key=lambda x: -x[1])
    return all_codes


def download_klines(ctx, codes_with_priority, max_requests):
    """下载K线数据, 最多使用 max_requests 个额度"""
    if not codes_with_priority:
        logger.info("没有待下载的合约")
        return

    # 加载已有数据
    existing_dfs = []
    if os.path.exists(DB_PATH):
        existing_dfs.append(pd.read_parquet(DB_PATH))

    downloaded = 0
    success = 0

    for code, priority, strike, expiry in codes_with_priority:
        if downloaded >= max_requests:
            break

        try:
            ret, data, _ = ctx.request_history_kline(
                code, start="2024-06-01",
                end=datetime.now().strftime("%Y-%m-%d"),
                ktype="K_DAY", max_count=1000)

            downloaded += 1

            if ret != RET_OK:
                if "额度不足" in str(data):
                    logger.warning("额度已用完, 停止下载")
                    break
                logger.warning(f"  {code} 失败: {str(data)[:80]}")
                time.sleep(1.05)
                continue

            if data.empty:
                time.sleep(1.05)
                continue

            parsed = parse_option_code(code)
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

            existing_dfs.append(df)
            success += 1
            logger.info(f"  [{downloaded}/{max_requests}] {code} "
                        f"K={strike:.0f} exp={expiry}: "
                        f"{len(data)} 行")

        except Exception as e:
            logger.warning(f"  {code} 异常: {e}")

        time.sleep(1.05)

    if success > 0:
        combined = pd.concat(existing_dfs, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["code", "date"], keep="last")
        combined.to_parquet(DB_PATH, index=False, engine="pyarrow")

        csv_path = DB_PATH.replace(".parquet", ".csv")
        combined.to_csv(csv_path, index=False)

        logger.info(f"数据库更新: {combined['code'].nunique()} 合约, "
                    f"{len(combined):,} 行")

    logger.info(f"本次: 请求={downloaded}, 成功={success}")


def main():
    logger.info("=" * 50)
    logger.info("智能K线下载 — 额度管理版")
    logger.info("=" * 50)

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

    try:
        # 1. 检查剩余额度
        # v3.7.199: moomoo SDK 返回 (RET_OK, (total, used, details)), 旧版直接拆 2 个 → tuple - int
        ret_code, data = ctx.get_history_kl_quota(get_detail=False)
        if isinstance(data, tuple) and len(data) >= 2:
            quota_total, quota_used = data[0], data[1]
        else:
            quota_total, quota_used = data, 0
        remaining = quota_total - quota_used
        logger.info(f"K线额度: 已用 {quota_used}/{quota_total}, "
                    f"剩余 {remaining}")

        if remaining <= 0:
            logger.info("额度为 0, 无法下载. 等待恢复.")
            return

        # 2. 获取优先合约列表
        logger.info("获取优先合约列表...")
        priority_codes = get_priority_contracts(ctx)
        logger.info(f"待下载: {len(priority_codes)} 个合约")

        if not priority_codes:
            logger.info("所有ATM合约已下载完毕")
            return

        # 显示前10个
        for code, pri, strike, exp in priority_codes[:10]:
            logger.info(f"  {code} K=${strike:.0f} "
                        f"exp={exp} pri={pri:.0f}")

        # 3. 下载 (使用可用额度)
        download_klines(ctx, priority_codes, max_requests=remaining)

    finally:
        ctx.close()


if __name__ == "__main__":
    main()
