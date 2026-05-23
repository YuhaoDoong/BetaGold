"""v3.7.220: sig_df 历史快照 — 冻结每日信号决策, 供 dashboard / ledger 回放.

设计原则:
  - 每个 (date, asset) 只存第一次评估的 sig_df 行 (append-only)
  - 后续 config 改动不影响历史行 (跟 ledger 冻结同源)
  - Dashboard marker palette 优先读这里, 找不到 fallback 实时 sig_df

存储: /Users/yhdong/Gold/data/sig_df_history.parquet
  index: 整数
  columns: date, asset, snapshot_at, + sig_df 全部列 (buy_signal, buy_type, ...)
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

HISTORY_PATH = "/Users/yhdong/Gold/data/sig_df_history.parquet"


def load_history() -> pd.DataFrame:
    if Path(HISTORY_PATH).exists():
        try:
            return pd.read_parquet(HISTORY_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_snapshot(asset: str, sig_df: pd.DataFrame,
                      evaluated_through: pd.Timestamp | None) -> int:
    """把 sig_df 中 > evaluated_through 的日期 append 到 history.
    去重: (date, asset) 已存在则保留旧 (freeze)."""
    if sig_df is None or len(sig_df) == 0:
        return 0
    now_iso = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    rows = []
    for d, row in sig_df.iterrows():
        d_norm = pd.Timestamp(d).normalize()
        if evaluated_through is not None and d_norm <= evaluated_through:
            continue
        r = row.to_dict()
        # 标准化日期 + asset
        r["date"] = d_norm
        r["asset"] = asset
        r["snapshot_at"] = now_iso
        rows.append(r)
    if not rows:
        return 0
    new_df = pd.DataFrame(rows)
    existing = load_history()
    if len(existing):
        merged = pd.concat([existing, new_df], ignore_index=True)
        # 第一次写入的胜出 (frozen)
        merged = merged.drop_duplicates(subset=["date", "asset"], keep="first")
        merged = merged.sort_values(["asset", "date"]).reset_index(drop=True)
    else:
        merged = new_df.sort_values(["asset", "date"]).reset_index(drop=True)
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(HISTORY_PATH, index=False)
    return len(rows)


def lookup(date: pd.Timestamp, asset: str, column: str = "buy_type"):
    """查 (date, asset) 的 column 值. 无快照返回 None."""
    history = load_history()
    if not len(history):
        return None
    d = pd.Timestamp(date).normalize()
    m = (history["date"] == d) & (history["asset"] == asset)
    sub = history[m]
    if not len(sub):
        return None
    return sub.iloc[0].get(column)


def lookup_row(date: pd.Timestamp, asset: str) -> dict | None:
    """查整行 sig_df. 无快照返回 None."""
    history = load_history()
    if not len(history):
        return None
    d = pd.Timestamp(date).normalize()
    m = (history["date"] == d) & (history["asset"] == asset)
    sub = history[m]
    if not len(sub):
        return None
    return sub.iloc[0].to_dict()
