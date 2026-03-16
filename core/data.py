"""数据加载模块 — 从 config.yaml 指定的路径读取所有数据."""

import os
import logging
from glob import glob
from datetime import datetime, timezone, timedelta

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# SGT/北京时区 (UTC+8)
_TZ_SGT = timezone(timedelta(hours=8))


def load_config(config_path: str = None) -> dict:
    """加载配置文件, 返回解析后的绝对路径."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["data_root"]
    resolved = {}
    for key, rel in cfg["paths"].items():
        resolved[key] = os.path.join(root, rel)
    cfg["resolved"] = resolved
    return cfg


def load_features(cfg: dict) -> pd.DataFrame:
    """加载特征矩阵."""
    return pd.read_parquet(cfg["resolved"]["features"])


def load_gld(cfg: dict) -> pd.DataFrame:
    """加载 GLD OHLCV."""
    return pd.read_csv(cfg["resolved"]["gld_csv"],
                       index_col=0, parse_dates=True)


def load_oos_predictions(cfg: dict) -> pd.DataFrame:
    """加载 DL Range OOS 预测."""
    return pd.read_parquet(cfg["resolved"]["oos_predictions"])


def load_gold_futures(cfg: dict) -> pd.DataFrame:
    """加载纽约黄金期货 (GC=F) OHLCV."""
    path = cfg["resolved"].get("gold_futures_csv")
    if path and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def load_usdcny(cfg: dict) -> pd.Series:
    """加载 USD/CNY 汇率. 返回 Close Series 或 None."""
    path = cfg["resolved"].get("usdcny_csv")
    if path and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["Close"]
    return None


def fetch_realtime_gold_fx():
    """获取实时金价和汇率 (via yfinance).

    Returns dict: {gc_price, usdcny, shfe_approx, timestamp}
    Returns None if fetch fails.
    """
    try:
        import yfinance as yf
        from datetime import datetime
        tickers = yf.Tickers("GC=F CNY=X")
        gc_info = tickers.tickers["GC=F"].fast_info
        cny_info = tickers.tickers["CNY=X"].fast_info
        gc_price = gc_info.get("lastPrice") or gc_info.get("previousClose")
        cny_rate = cny_info.get("lastPrice") or cny_info.get("previousClose")
        if gc_price and cny_rate:
            return {
                "gc_price": float(gc_price),
                "usdcny": float(cny_rate),
                "shfe_approx": float(gc_price) * float(cny_rate) / 31.1035,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
    except Exception:
        pass
    return None


def get_today_sgt():
    """返回 SGT (UTC+8) 的今日日期."""
    return datetime.now(_TZ_SGT).date()


def auto_refresh_market_data(cfg: dict):
    """检测市场数据是否过期, 自动下载最新数据.

    检查 GLD, 黄金期货, USD/CNY 三个 CSV 的最后日期,
    如果落后于今天 (SGT) 且是交易日, 则用 yfinance 下载增量数据并追加.

    Returns: list of (ticker, status_message)
    """
    today = get_today_sgt()
    results = []

    updates = [
        ("gld_csv", "GLD", "GLD"),
        ("gold_futures_csv", "黄金期货", "GC=F"),
        ("usdcny_csv", "USD/CNY", "CNY=X"),
    ]

    for cfg_key, label, yf_ticker in updates:
        path = cfg["resolved"].get(cfg_key)
        if not path or not os.path.exists(path):
            results.append((label, "文件不存在, 跳过"))
            continue

        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            last_date = existing.index[-1].date()

            # 如果最后日期 >= 上一个交易日, 无需更新
            # 周末: 周六/日 → 周五是最后交易日
            ref = today
            wd = ref.weekday()
            if wd == 5:  # Saturday
                last_bday = ref - timedelta(days=1)
            elif wd == 6:  # Sunday
                last_bday = ref - timedelta(days=2)
            else:
                last_bday = ref

            if last_date >= last_bday:
                results.append((label, f"已是最新 ({last_date})"))
                continue

            # 下载增量数据
            import yfinance as yf
            start = last_date + timedelta(days=1)
            ticker = yf.Ticker(yf_ticker)
            new_data = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"))

            if new_data is None or len(new_data) == 0:
                results.append((label, f"无新数据 (最新 {last_date})"))
                continue

            # 统一列名 (yfinance 返回 Open/High/Low/Close/Volume)
            new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
            new_data.index.name = "Date"
            cols_keep = [c for c in ["Close", "High", "Low", "Open", "Volume"]
                         if c in new_data.columns]
            new_data = new_data[cols_keep]

            # 去重: 只保留比 existing 更新的日期
            new_data = new_data[new_data.index > existing.index[-1]]
            if len(new_data) == 0:
                results.append((label, f"无新数据 (最新 {last_date})"))
                continue

            # 追加并保存
            combined = pd.concat([existing, new_data])
            combined.to_csv(path)
            new_last = combined.index[-1].date()
            results.append(
                (label, f"更新 {last_date} → {new_last} (+{len(new_data)}行)"))
            logger.info("Updated %s: %s → %s (+%d rows)",
                        label, last_date, new_last, len(new_data))

        except Exception as e:
            results.append((label, f"更新失败: {e}"))
            logger.warning("Failed to update %s: %s", label, e)

    return results


def load_latest_eod_snapshot(cfg: dict):
    """加载最新 EOD 期权快照. 返回 (df, date_str) 或 (None, None)."""
    snap_dir = cfg["resolved"]["eod_snapshots"]
    if not os.path.isdir(snap_dir):
        return None, None
    snaps = sorted(glob(os.path.join(snap_dir, "202*", "eod_full.parquet")))
    if not snaps:
        return None, None
    latest = snaps[-1]
    snap_date = os.path.basename(os.path.dirname(latest))
    return pd.read_parquet(latest), snap_date


def load_all_eod_snapshots(cfg: dict):
    """加载所有 EOD 快照. 返回 {pd.Timestamp: DataFrame}."""
    snap_dir = cfg["resolved"]["eod_snapshots"]
    if not os.path.isdir(snap_dir):
        return {}
    snaps = sorted(glob(os.path.join(snap_dir, "202*", "eod_full.parquet")))
    result = {}
    for path in snaps:
        date_str = os.path.basename(os.path.dirname(path))
        try:
            ts = pd.Timestamp(date_str)
            result[ts] = pd.read_parquet(path)
        except Exception:
            pass
    return result
