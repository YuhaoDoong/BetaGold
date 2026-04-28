"""期货 P&L 计算 — 与期权对比的基础设施.

期货特性:
  - 线性 delta (价格涨 1% 赚 1%)
  - 无 vega / gamma / theta 损耗
  - P&L = 收盘 ret_5d (无止损时)
  - 加止损后 P&L = max(ret_5d, -stop_pct) if max_down > stop_pct else ret_5d

实证 (近 5y, BUY CALL 信号下):
  - 期货无止损: 95% 胜率, Sharpe 1.14
  - 期货 + 3% 止损: 96% 胜率, Sharpe 1.16

期货代替期权的适用边界:
  - BUY CALL 信号 (RV < 0.50): 期货 96% > 期权 73%, 推荐用期货
  - SELL PUT 信号 (RV > 0.85): 期权 100% > 期货 68%, 必须用期权
"""
from typing import Optional


def futures_pnl(ret_5d: Optional[float]) -> Optional[float]:
    """期货多头无止损 P&L (线性).

    Args:
        ret_5d: 5天后总收益 %
    """
    return ret_5d


def futures_pnl_with_stop(ret_5d: Optional[float],
                            move_down: Optional[float],
                            stop_pct: float = 3.0) -> Optional[float]:
    """期货多头 + 硬止损 P&L.

    若 5 天内日内最低跌幅 > stop_pct, 视为先触发止损; 否则按收盘 ret_5d.

    Args:
        ret_5d: 5天后总收益 %
        move_down: 持仓期最大下行 % (正值, 高 = 跌多)
        stop_pct: 止损阈值 (默认 3%)
    """
    if ret_5d is None or move_down is None:
        return None
    if move_down > stop_pct:
        return -stop_pct
    return ret_5d


def futures_win(ret_5d: Optional[float],
                threshold: float = 0.0) -> Optional[bool]:
    """期货多头胜率."""
    if ret_5d is None:
        return None
    return ret_5d > threshold


def futures_stop_win(ret_5d: Optional[float],
                       move_down: Optional[float],
                       stop_pct: float = 3.0) -> Optional[bool]:
    """期货 + 止损 胜率."""
    pnl = futures_pnl_with_stop(ret_5d, move_down, stop_pct)
    if pnl is None:
        return None
    return pnl > 0
