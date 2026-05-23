"""v3.7.221: Dashboard 可视化快照 — 每次刷新覆盖, 反映当时图上"画了什么".

设计原则:
  - 每个 asset 一份 JSON, 每次 dashboard 渲染时覆盖
  - 记录 "marker 实际 rendering 的状态" (不是模型 raw 输出)
  - 用户事后能 diff "我看到的" vs "数据应该是的"

文件: /Users/yhdong/Gold/data/dashboard_snapshots/{asset}.json
  schema:
    timestamp: ISO SGT (本次渲染时间)
    asset: GLD / SLV
    viz_window: [start, end]
    main_chart_markers: [{date, price, strategy, color, marker_shape, source}]
    sub_chart_markers: [{trigger_time, price, futures_price, side, type, color, marker_shape, in_us_session}]
    sig_df_window: list of dicts (sig_df 行, viz 窗口内)
    ledger_in_window: list of ledger 行 (viz 窗口内)
    unified_viz_window: list of unified chosen 行
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pandas as pd

SNAPSHOT_DIR = Path("/Users/yhdong/Gold/data/dashboard_snapshots")


def _ts() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def _to_native(v):
    """numpy / pandas → 原生 Python (JSON 可序列化)."""
    if v is None: return None
    if isinstance(v, (pd.Timestamp,)): return v.isoformat()
    if isinstance(v, (bool,)): return bool(v)
    try:
        import numpy as np
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)):
            if pd.isna(v): return None
            return float(v)
    except Exception: pass
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v


def _row_to_dict(row) -> dict:
    if hasattr(row, "to_dict"): row = row.to_dict()
    return {k: _to_native(v) for k, v in row.items()}


def save(asset: str,
          viz_dates: pd.DatetimeIndex,
          sig_df: pd.DataFrame,
          ledger_rows: list,
          intraday_log_window: pd.DataFrame,
          unified_viz: pd.DataFrame,
          main_chart_markers: list = None,
          sub_chart_markers: list = None) -> str:
    """覆盖式写入 asset.json. 返回路径."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{asset}.json"

    # sig_df window subset
    sig_window = []
    if sig_df is not None and len(sig_df) and len(viz_dates):
        d0, d1 = viz_dates[0], viz_dates[-1]
        sub = sig_df[(sig_df.index >= d0) & (sig_df.index <= d1)]
        for d, row in sub.iterrows():
            r = _row_to_dict(row); r["date"] = pd.Timestamp(d).isoformat()
            sig_window.append(r)

    # ledger window subset
    ledger_window = []
    if ledger_rows and len(viz_dates):
        d0, d1 = viz_dates[0].normalize(), viz_dates[-1].normalize()
        for r in ledger_rows:
            try:
                sd = pd.Timestamp(r.get("signal_date", "")).normalize()
                if d0 <= sd <= d1:
                    ledger_window.append(r)
            except Exception:
                pass

    # unified_viz window subset
    uni_window = []
    if unified_viz is not None and len(unified_viz):
        for d, row in unified_viz.iterrows():
            d_norm = pd.Timestamp(d).normalize()
            if len(viz_dates) and (d_norm < viz_dates[0] or d_norm > viz_dates[-1]):
                continue
            r = _row_to_dict(row); r["date"] = d_norm.isoformat()
            uni_window.append(r)

    # intraday log window
    intra_window = []
    if intraday_log_window is not None and len(intraday_log_window):
        for _, row in intraday_log_window.iterrows():
            intra_window.append(_row_to_dict(row))

    payload = {
        "timestamp": _ts(),
        "asset": asset,
        "viz_window": [viz_dates[0].isoformat() if len(viz_dates) else None,
                        viz_dates[-1].isoformat() if len(viz_dates) else None],
        "main_chart_markers": main_chart_markers or [],
        "sub_chart_markers": sub_chart_markers or [],
        "sig_df_window": sig_window,
        "ledger_in_window": ledger_window,
        "unified_viz_window": uni_window,
        "intraday_log_window": intra_window,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)
    return str(path)
