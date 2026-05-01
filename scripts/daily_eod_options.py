"""每日 EOD 期权快照采集 (v3.7.52, B 方案).

目的:
  补 yfinance 之外的本地累积 — 用于:
  1. 已过期期权的历史 (yfinance 删除)
  2. 远 OTM strike (yfinance 没数据)
  3. IC 4 腿验证

工作流:
  1. 拉 GLD/SLV 全部活跃 expiry 列表
  2. 对每个 expiry, 拉 ATM ±10 strikes 的当日 OHLC
  3. Append 到 data/raw/options_history/kline_db/all_klines.parquet
  4. 同时保存当日 full snapshot 到 data/raw/options_history/<date>/eod_full.csv

调度建议 (cron):
  # SGT 04:30 (US 收盘后 30min)
  30 4 * * * cd /Users/yhdong/GoldDash && \
    conda run -n gold python scripts/daily_eod_options.py

用法:
  python scripts/daily_eod_options.py             # GLD+SLV
  python scripts/daily_eod_options.py --asset GLD # 单资产
  python scripts/daily_eod_options.py --strikes 20  # 拉 ATM ±20 strikes
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


KLINE_DB_PATH = ("/Users/yhdong/Gold/data/raw/options_history/"
                  "kline_db/all_klines.parquet")
SNAPSHOT_DIR = "/Users/yhdong/Gold/data/raw/options_history"


def load_kline_db():
    if os.path.exists(KLINE_DB_PATH):
        return pd.read_parquet(KLINE_DB_PATH)
    return pd.DataFrame(columns=["date","open","high","low","close","volume",
                                    "code","strike","expiry","option_type",
                                    "dte_at_date"])


def save_kline_db(df):
    os.makedirs(os.path.dirname(KLINE_DB_PATH), exist_ok=True)
    # v3.7.56: dtype 一致化 (避免 pyarrow mixed-type 报错)
    df = df.copy()
    df["date"] = df["date"].astype(str).str[:10]  # YYYY-MM-DD 字符串
    for c in ["open","high","low","close","volume","strike","dte_at_date"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["code","expiry","option_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    df.to_parquet(KLINE_DB_PATH, index=False)
    csv_path = KLINE_DB_PATH.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)


def fetch_spot(ticker: str) -> float:
    """获取当前 spot."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="5d")
    if df is None or len(df) == 0:
        raise RuntimeError(f"无法拉 {ticker} spot")
    return float(df["Close"].iloc[-1])


def fetch_active_expiries(ticker: str) -> list:
    import yfinance as yf
    return list(yf.Ticker(ticker).options)


def fetch_strike_history(ticker: str, expiry: str, strike: float,
                            opt_type: str, period: str = "2y"):
    """拉单 strike 全部历史."""
    import yfinance as yf
    yymmdd = pd.Timestamp(expiry).strftime("%y%m%d")
    sym = f"{ticker}{yymmdd}{opt_type}{int(strike*1000):08d}"
    try:
        df = yf.Ticker(sym).history(period=period)
        if df is None or len(df) == 0:
            return None, sym
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df, sym
    except Exception as e:
        return None, sym


def collect_asset(ticker: str, n_strikes: int = 10,
                    expiries_max: int = 8):
    """对单资产拉所有活跃 expiry × ATM ± n_strikes 的历史.

    叠加到 kline_db (去重保留最新).
    """
    spot = fetch_spot(ticker)
    print(f"\n{ticker} spot: ${spot:.2f}")
    expiries = fetch_active_expiries(ticker)
    print(f"  活跃 expiry: {len(expiries)} 个")

    # 限制 expiry 数量 (优先月度第三周五 + LEAPS)
    today = pd.Timestamp.now().normalize()
    selected = []
    for e in sorted(expiries):
        d = pd.Timestamp(e)
        if d <= today:
            continue
        dte = (d - today).days
        is_third_friday = (d.weekday() == 4 and 15 <= d.day <= 21)
        # 月度 (第三周五) 或 DTE > 90 (季度+LEAPS) 优先
        if is_third_friday or dte > 90:
            selected.append(e)
        if len(selected) >= expiries_max:
            break
    print(f"  采集 expiry: {len(selected)} 个")

    # strike 步长 (GLD $5, SLV $1)
    if ticker == "GLD":
        step = 5
        base = round(spot / step) * step
    else:  # SLV
        step = 1
        base = round(spot)

    rows = []
    for expiry in selected:
        dte_now = (pd.Timestamp(expiry) - today).days
        for offset in range(-n_strikes, n_strikes + 1):
            strike = base + offset * step
            if strike <= 0:
                continue
            for opt_type in ["C", "P"]:
                df, sym = fetch_strike_history(ticker, expiry, strike, opt_type)
                if df is None or len(df) < 5:
                    continue
                # 转 kline_db schema
                # kline_db code: US.<TIC><YYMMDD><C/P><strike int 5位>
                yymmdd = pd.Timestamp(expiry).strftime("%y%m%d")
                strike_5 = int(strike * 1000)
                mm_code = f"US.{ticker}{yymmdd}{opt_type}{strike_5}"
                full_type = "CALL" if opt_type == "C" else "PUT"
                for d, row in df.iterrows():
                    dte_at_d = (pd.Timestamp(expiry) - d).days
                    rows.append({
                        "date": d.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row.get("Volume", 0)),
                        "code": mm_code,
                        "strike": float(strike),
                        "expiry": expiry,
                        "option_type": full_type,
                        "dte_at_date": dte_at_d,
                    })
        print(f"  {expiry} (DTE {dte_now}): 累计 {len(rows)} 行")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None,
                         help="GLD / SLV, 默认两个都跑")
    parser.add_argument("--strikes", type=int, default=10,
                         help="ATM ± n_strikes (默认 10)")
    parser.add_argument("--expiries-max", type=int, default=8)
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ["GLD", "SLV"]
    print(f"=== 每日 EOD 期权采集 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"资产: {assets}, ATM ±{args.strikes} strikes, "
          f"top {args.expiries_max} expiries")

    db = load_kline_db()
    print(f"\n现有 kline_db: {len(db)} 行")

    new_rows_total = 0
    for asset in assets:
        try:
            new_df = collect_asset(asset, args.strikes, args.expiries_max)
            if len(new_df) == 0:
                print(f"\n{asset}: 无新数据")
                continue
            # 合并 (去重: code+date 联合主键, 新值覆盖)
            db = pd.concat([db, new_df], ignore_index=True)
            db = db.drop_duplicates(subset=["code", "date"], keep="last")
            new_rows_total += len(new_df)
            print(f"\n{asset}: 新增 {len(new_df)} 行 (合并去重后总 {len(db)})")
        except Exception as e:
            print(f"{asset}: 失败 — {e}")

    save_kline_db(db)
    print(f"\n保存 → {KLINE_DB_PATH}, 共 {len(db)} 行 (本次 +{new_rows_total})")
    print(f"覆盖资产: {db['code'].str.extract(r'US.(GLD|SLV)')[0].value_counts().to_dict()}")
    print(f"日期范围: {db['date'].min()} → {db['date'].max()}")


if __name__ == "__main__":
    main()
