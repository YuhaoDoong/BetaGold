"""持仓 ledger — 信号触发时快照固定, 跨 dashboard reload 一致.

用户洞察: 触发信号时要快照价格 + 策略 + 成本, 后续不变.
之前问题: 每次 dashboard 加载重算 entry pricing, CSV 数据微变就显示不一致.

Schema (parquet):
  signal_date    (date)        — 信号日 (UTC normalized)
  asset          (str)         — GLD / SLV
  strategy       (str)         — BC / SP / STRADDLE / SHORT_VOL / FUTURES
  entry_spot     (float)       — 入场 underlying spot (ETF 价)
  entry_credit   (float)       — net credit / debit (期权) 或 entry spot (期货)
  legs_json      (str)         — JSON list of [label, code, strike, qty]
  entry_leg_prices_json (str)  — JSON [[label, price], ...]
  source         (str)         — 显示用 (e.g. "IC -P$64/+P$62 -C$68/+C$71 (05/15)")
  entry_timestamp (datetime)   — 触发时刻 (ET)
  max_risk       (float)       — 单笔 margin / premium

  exit_date      (date)        — 平仓日 (NaT if OPEN)
  exit_value     (float)       — 平仓价 (NaN if OPEN)
  exit_reason    (str)         — TP/SL/expiry/timeout/...
  exit_pnl_pct   (float)       — 最终 PnL %

操作:
  upsert_position()  — 信号触发时写入 (key=signal_date+asset+strategy)
  load_positions()   — 加载 (用于 dashboard 显示)
  mark_closed()      — 平仓时更新 exit 字段
"""
from __future__ import annotations
import json
import os
import pandas as pd
from typing import Optional

LEDGER_PATH = "/Users/yhdong/Gold/data/positions_ledger.parquet"
LEDGER_COLS = [
    "signal_date", "asset", "strategy",
    "entry_spot", "entry_credit", "legs_json", "entry_leg_prices_json",
    "source", "entry_timestamp", "max_risk",
    "exit_date", "exit_value", "exit_reason", "exit_pnl_pct",
]


def load_ledger() -> pd.DataFrame:
    if not os.path.exists(LEDGER_PATH):
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_parquet(LEDGER_PATH)
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    if "entry_timestamp" in df.columns:
        df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"])
    return df


def save_ledger(df: pd.DataFrame):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    df.to_parquet(LEDGER_PATH, index=False)


def position_key(signal_date, asset, strategy):
    return (pd.Timestamp(signal_date).normalize(), asset, strategy)


def upsert_position(signal_date, asset: str, strategy: str,
                       entry_pricing: dict, entry_timestamp=None,
                       max_risk: float = 0.0,
                       force_update: bool = False) -> bool:
    """信号触发时写入 ledger. 已存在则保留旧 entry (snapshot lock).

    Returns: True if newly inserted, False if already existed.
    """
    df = load_ledger()
    key = position_key(signal_date, asset, strategy)
    if not df.empty:
        existing = df[
            (df["signal_date"] == key[0]) &
            (df["asset"] == key[1]) &
            (df["strategy"] == key[2])
        ]
        if len(existing) and not force_update:
            return False  # 已存在, snapshot 锁定不变
    new_row = {
        "signal_date": pd.Timestamp(signal_date).normalize(),
        "asset": asset,
        "strategy": strategy,
        "entry_spot": float(entry_pricing.get("daily_close_price", 0)
                              or entry_pricing.get("entry_price", 0)),
        "entry_credit": float(entry_pricing.get("entry_price", 0)),
        "legs_json": json.dumps([list(l) for l in entry_pricing.get("legs", [])]),
        "entry_leg_prices_json": json.dumps(
            [list(p) for p in entry_pricing.get("leg_prices", [])]),
        "source": entry_pricing.get("source", ""),
        "entry_timestamp": pd.Timestamp(entry_timestamp) if entry_timestamp else
                            pd.Timestamp(signal_date).normalize() + pd.Timedelta(hours=9, minutes=30),
        "max_risk": float(max_risk),
        "exit_date": pd.NaT,
        "exit_value": float("nan"),
        "exit_reason": "",
        "exit_pnl_pct": float("nan"),
    }
    if df.empty:
        df = pd.DataFrame([new_row])
    else:
        # 删除旧 (force_update 模式) 后 append
        mask = ((df["signal_date"] == key[0])
                & (df["asset"] == key[1])
                & (df["strategy"] == key[2]))
        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(df)
    return True


def mark_closed(signal_date, asset: str, strategy: str,
                  exit_date, exit_value: float, exit_reason: str,
                  exit_pnl_pct: float):
    df = load_ledger()
    if df.empty: return False
    key = position_key(signal_date, asset, strategy)
    mask = ((df["signal_date"] == key[0])
            & (df["asset"] == key[1])
            & (df["strategy"] == key[2]))
    if not mask.any(): return False
    df.loc[mask, "exit_date"] = pd.Timestamp(exit_date)
    df.loc[mask, "exit_value"] = float(exit_value)
    df.loc[mask, "exit_reason"] = str(exit_reason)
    df.loc[mask, "exit_pnl_pct"] = float(exit_pnl_pct)
    save_ledger(df)
    return True


def get_position(signal_date, asset: str, strategy: str) -> Optional[dict]:
    """取出 snapshotted entry pricing (用于 dashboard 显示, 替代重算)."""
    df = load_ledger()
    if df.empty: return None
    key = position_key(signal_date, asset, strategy)
    row = df[
        (df["signal_date"] == key[0]) &
        (df["asset"] == key[1]) &
        (df["strategy"] == key[2])
    ]
    if not len(row): return None
    r = row.iloc[0]
    legs = [tuple(l) for l in json.loads(r["legs_json"])]
    leg_prices = [tuple(p) for p in json.loads(r["entry_leg_prices_json"])]
    return {
        "entry_price": float(r["entry_credit"]),
        "legs": legs,
        "leg_prices": leg_prices,
        "source": r["source"],
        "entry_spot": float(r["entry_spot"]),
        "entry_timestamp": r["entry_timestamp"],
        "max_risk": float(r["max_risk"]),
        "is_closed": pd.notna(r["exit_date"]),
        "exit_date": r["exit_date"],
        "exit_value": float(r["exit_value"]) if pd.notna(r["exit_value"]) else None,
        "exit_reason": r["exit_reason"],
        "exit_pnl_pct": float(r["exit_pnl_pct"]) if pd.notna(r["exit_pnl_pct"]) else None,
    }


def list_open_positions(asset: Optional[str] = None) -> pd.DataFrame:
    """所有 OPEN 持仓 (exit_date is NaT)."""
    df = load_ledger()
    if df.empty: return df
    df = df[df["exit_date"].isna()]
    if asset is not None:
        df = df[df["asset"] == asset]
    return df


def list_closed_positions(asset: Optional[str] = None,
                            window_days: int = 30) -> pd.DataFrame:
    """已平仓持仓 (近 N 天)."""
    df = load_ledger()
    if df.empty: return df
    df = df[df["exit_date"].notna()]
    if asset is not None:
        df = df[df["asset"] == asset]
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
    df = df[df["exit_date"] >= cutoff]
    return df
