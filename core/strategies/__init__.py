"""策略模块 — 按工具/方向分组, 方便后续修改替换.

模块映射:
  win_metrics.py  — 各策略胜率定义 (按 vega/delta 实际盈亏)
  futures_pnl.py  — 期货 P&L 计算 (与期权对比基础)

策略实现仍在 core/ 根目录 (向后兼容):
  signals_v2.py   — 方向性 (BUY CALL / SELL PUT) + RV 过滤
  events.py       — STRADDLE / SHORT_VOL Iron Condor + 事件日历
  strategy_selector.py — 三方竞争统一策略
"""
from .win_metrics import (
    compute_win,
    compute_dir_win,
    compute_vol_win,
    compute_futures_win,
    WIN_DEFINITIONS,
)
from .futures_pnl import (
    futures_pnl,
    futures_pnl_with_stop,
    futures_win,
    futures_stop_win,
)

__all__ = [
    "compute_win",
    "compute_dir_win",
    "compute_vol_win",
    "compute_futures_win",
    "WIN_DEFINITIONS",
    "futures_pnl",
    "futures_pnl_with_stop",
    "futures_win",
    "futures_stop_win",
]
