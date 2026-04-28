"""胜率定义 — 按各策略 vega/delta 实际盈亏特性.

设计原则:
  胜率不应固定阈值, 应反映该策略真实的 P&L 模型.
  全部用动态 sigma_pct = RV × √(hold_days/252) 自适应当时波动环境.

策略 vega/delta 矩阵:
  BUY CALL  long delta + long vega + long gamma  (低 RV 入场, 赌移动)
  SELL PUT  long delta + short vega + short gamma (高 RV 入场, 收 IV)
  STRADDLE  neutral delta + long vega + long gamma (双向赌大动)
  SHORT_VOL neutral delta + short vega + short gamma (赌静止)
  期货多头  linear delta, vega=0, gamma=0, theta=0 (线性方向性)

胜率定义:
  BUY CALL: max_up > 1σ
            (必须真涨且超过 IV cost, 横盘亏 theta+IV crush)
  SELL PUT: max_down < 1σ
            (跌不破 1σ 短 put strike, 横盘+上涨都赢)
  STRADDLE: max_move > 1σ
            (双向移动 > 收入的 premium)
  SHORT_VOL: max_move < strike_sigma × σ (默认 1.6σ short strike)
            (波动留在 IC 短腿内, 留 credit)
  期货多头: ret_5d > 0
            (无 IV cost, 任何正向收盘都赢)

所有阈值都用动态 sigma_pct, 自动适应当时 RV 水平.
"""
from typing import Optional


def _safe_compare(left, right, op):
    """None 安全的比较."""
    if left is None or right is None:
        return None
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    raise ValueError(f"unsupported op {op}")


def compute_dir_win(dir_type: str, move_up: Optional[float],
                    move_down: Optional[float], sigma_pct: float
                    ) -> Optional[bool]:
    """方向性策略胜率 (BUY CALL / SELL PUT).

    Args:
        dir_type: 'BUY CALL' 或 'SELL PUT'
        move_up: 持仓期最大上行 % (相对入场价)
        move_down: 持仓期最大下行 % (相对入场价, 正值)
        sigma_pct: 1σ 波动 % (RV-based)
    """
    if dir_type == "BUY CALL":
        return _safe_compare(move_up, sigma_pct, ">")
    if dir_type == "SELL PUT":
        return _safe_compare(move_down, sigma_pct, "<")
    return None


def compute_vol_win(vol_type: str, max_move: Optional[float],
                    sigma_pct: float, ic_strike_sigma: float = 1.6,
                    iv_crush_factor: float = 0.0,
                    ) -> Optional[bool]:
    """波动率策略胜率 (STRADDLE / SHORT_VOL).

    Args:
        vol_type: 'STRADDLE' 或 'SHORT_VOL'
        max_move: 持仓期最大单向移动 % (max(move_up, move_down))
        sigma_pct: 1σ 波动 %
        ic_strike_sigma: SHORT_VOL IC 短腿距离倍数 (默认 1.6)
        iv_crush_factor: 跨事件后 IV crush 损失比例 (long vol 累加)
            FOMC ~0.30, NFP ~0.15, OPEX ~0.10. 多事件相加.
            STRADDLE 是 long vega → 抬高 win 阈值 (赢条件更难)
            SHORT_VOL 是 short vega → 降低 win 阈值 (IV crush 帮短 vol)
    """
    if vol_type == "STRADDLE":
        # Long vol: 真实 cost = sigma + iv_crush_loss → win 阈值上调
        return _safe_compare(max_move, sigma_pct * (1 + iv_crush_factor), ">")
    if vol_type == "SHORT_VOL":
        # Short vol: IV crush 帮我们, 短腿空间相当于变宽
        return _safe_compare(max_move,
                              sigma_pct * ic_strike_sigma * (1 + iv_crush_factor * 0.5),
                              "<")
    return None


def compute_futures_win(ret_5d: Optional[float],
                          threshold: float = 0.0) -> Optional[bool]:
    """期货多头胜率 (线性 P&L, 无 IV cost).

    Args:
        ret_5d: 5天后总收益 %
        threshold: 胜利阈值 (默认 0%, 即任何正收益都赢)
    """
    return _safe_compare(ret_5d, threshold, ">")


def compute_win(chosen: str, *,
                ret_5d: Optional[float] = None,
                move_up: Optional[float] = None,
                move_down: Optional[float] = None,
                max_move: Optional[float] = None,
                sigma_pct: float = 0.0,
                ic_strike_sigma: float = 1.6) -> Optional[bool]:
    """统一胜率接口 — 根据 chosen 类型自动分发.

    支持单策略和 MIXED 组合 (用 ' + ' 分隔). MIXED 任一胜即赢.
    """
    if chosen == "EXIT":
        return ret_5d < 3 if ret_5d is not None else None

    if " + " in chosen:
        base_dir, base_vol = chosen.split(" + ")
        dw = compute_dir_win(base_dir, move_up, move_down, sigma_pct)
        vw = compute_vol_win(base_vol, max_move, sigma_pct, ic_strike_sigma)
        if dw is None and vw is None:
            return None
        return bool(dw) or bool(vw)

    if chosen in ("BUY CALL", "SELL PUT"):
        return compute_dir_win(chosen, move_up, move_down, sigma_pct)

    if chosen in ("STRADDLE", "SHORT_VOL"):
        return compute_vol_win(chosen, max_move, sigma_pct, ic_strike_sigma)

    # 兜底: 5天后未跌超 3% 算赢
    return ret_5d > -3 if ret_5d is not None else None


# 文档化的策略-胜率映射 (供 README / 测试参考)
WIN_DEFINITIONS = {
    "BUY CALL": "max_up > 1σ (RV-动态)",
    "SELL PUT": "max_down < 1σ",
    "STRADDLE": "max_move > 1σ",
    "SHORT_VOL": "max_move < 1.6σ (IC 短腿距离)",
    "FUTURES_LONG": "ret_5d > 0",
    "FUTURES_LONG_STOP": "ret_5d > 0 且 max_down < 3% (止损未触发)",
    "EXIT": "ret_5d < 3% (5天后未涨超 3%)",
    "MIXED": "方向性赢 OR 波动率赢, 任一胜即赢",
}
