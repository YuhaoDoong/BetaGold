"""v3.7.187 被过滤信号 log — 记录 raw 触发但 filter 阻拦的信号.

设计:
  pass-filter 信号 → intraday_signal_log.parquet (旧)
  filtered-blocked → filtered_signal_log.parquet (新)

显示规则 (用户明令):
  今日: 显示 pass-filter + filtered-blocked (双 source)
  历史: 仅显示 pass-filter (不显示 blocked)

字段:
  date, asset, candidate_strategy, filter_reason, raw_trigger_price,
  raw_trigger_time, detect_source (daily/intraday)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

LOG_PATH = "/Users/yhdong/Gold/data/filtered_signal_log.parquet"

SCHEMA = [
    "date", "asset", "candidate_strategy", "filter_reason",
    "raw_trigger_price", "raw_trigger_time", "detect_source",
]


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=SCHEMA)
    try:
        return pd.read_parquet(LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=SCHEMA)


def append_log(rows: list[dict]) -> int:
    """rows: list of dict with SCHEMA keys. 自动 dedupe (date, asset, candidate, reason)."""
    if not rows: return 0
    df_new = pd.DataFrame(rows)
    df_new["date"] = pd.to_datetime(df_new["date"])
    df_old = load_log()
    if not df_old.empty:
        df_old["date"] = pd.to_datetime(df_old["date"])
    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "asset", "candidate_strategy", "filter_reason"],
        keep="last")
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOG_PATH)
    return len(df_new)


def today_filtered(today_dt: pd.Timestamp, asset: str = None) -> pd.DataFrame:
    """返回今日被过滤记录."""
    df = load_log()
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["date"].dt.normalize() == pd.Timestamp(today_dt).normalize()]
    if asset:
        sub = sub[sub["asset"] == asset]
    return sub
