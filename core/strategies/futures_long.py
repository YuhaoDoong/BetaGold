"""期货多头独立策略模块 (Binance XAUUSDT / GC=F perp).

设计原则:
  1. leverage 参数化 (Binance XAUUSDT max=20, dYdX 50, Bybit 多至 100)
  2. 爆仓价感知 SL — 高杠杆时自动收紧止损 (避免实际爆仓)
  3. TP/SL/hold 三参数 grid 后定值 (v3.7.129)
  4. 资金费 + 双边 taker fee 计入净 PnL

爆仓公式 (cross margin, 简化):
  long_liq = entry × (1 - 1/lev + mm_rate)
  XAUUSDT mm_rate = 0.005 (Bracket 1, < $50k notional)

  lev=20:  跌 4.5% 爆仓
  lev=50:  跌 1.5% 爆仓
  lev=100: 跌 0.5% 爆仓 (基本必爆)
  lev=10:  跌 9.5% 爆仓
  lev=5:   跌 19.5% 爆仓
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FuturesConfig:
    """期货策略参数 (paper_positions + backtest 共用)."""
    leverage: int = 20                   # 杠杆 (Binance XAUUSDT 上限 20)
    maintenance_margin_rate: float = 0.005  # MM rate (5y XAUUSDT 实测)

    # 退出参数 (v3.7.129 grid 最优)
    tp_pct: float = 8.0                  # spot % 止盈
    sl_pct: float = 5.0                  # spot % 止损 (相对 entry)
    hold_max_days: int = 15              # 最长持仓

    # 爆仓感知 SL — 距爆仓价 buffer
    auto_sl_from_liq: bool = True        # True=自动收紧 SL 到爆仓 - buffer
    liq_buffer_pct: float = 1.0          # 距爆仓 buffer (%)

    # 费用
    taker_fee: float = 0.0005            # 单边 0.05% (Binance regular)
    funding_rate_8h: float = 0.0001      # 平均 (实测可拉)


def liquidation_distance_pct(cfg: FuturesConfig, side: str = "long") -> float:
    """相对 entry 的爆仓价距离 (%, long 为负, short 为正)."""
    base = 1.0 / cfg.leverage - cfg.maintenance_margin_rate
    return -base * 100 if side == "long" else base * 100


def effective_sl_pct(cfg: FuturesConfig, side: str = "long") -> float:
    """实际止损 % (考虑爆仓: 取 sl_pct 与 liq_buffer 的较紧者).

    用户配置 sl=5% 但 lev=50 (爆仓 1.5%) → 自动收紧到 0.5% SL.
    """
    if not cfg.auto_sl_from_liq:
        return cfg.sl_pct
    liq_dist = abs(liquidation_distance_pct(cfg, side))
    safe_sl = liq_dist - cfg.liq_buffer_pct
    return min(cfg.sl_pct, max(0.5, safe_sl))  # 至少 0.5% (避免 lev=100 时 SL=0)


def simulate_long_position(entry_d: pd.Timestamp,
                              entry_spot: float,
                              ohlc: pd.DataFrame,
                              today: pd.Timestamp,
                              cfg: FuturesConfig = None,
                              live_spot: float = None,
                              live_high: float = None,
                              live_low: float = None) -> dict:
    """模拟期货多头持仓 — 4 层退出: 爆仓 / SL / TP / Timeout.

    v3.7.134: 期货 24h 可交易 (Binance XAUUSDT / GC=F COMEX 23h).
      live_spot: 当前实时价 (用于 MTM, 不再用上日 close)
      live_high/low: 今日盘中 high/low (用于检查 SL/TP 是否已被触发)
                     若 None, 用 live_spot 当 high=low (保守 — 仅 close 检测)
    时间序: bar 内最坏先发 (保守).
    """
    if cfg is None: cfg = FuturesConfig()
    sl = effective_sl_pct(cfg)
    liq_pct = liquidation_distance_pct(cfg, "long")  # 负值
    liq_price = entry_spot * (1 + liq_pct / 100)

    later = ohlc.index[ohlc.index > entry_d]
    if not len(later):
        # 即使没历史 daily, 仍可用 live spot 检查 (entry 同日紧急触发)
        if live_spot is not None:
            return _check_live(entry_spot, live_spot, live_high, live_low,
                                hold=0, cfg=cfg, sl=sl, liq_pct=liq_pct,
                                liq_price=liq_price, today=today)
        return {"closed": False, "reason": "no later data"}

    hold = 0
    for d in later:
        if pd.Timestamp(d) > today: break
        hold += 1
        H = float(ohlc.loc[d, "High"])
        L = float(ohlc.loc[d, "Low"])
        C = float(ohlc.loc[d, "Close"])
        rL = (L / entry_spot - 1) * 100
        rH = (H / entry_spot - 1) * 100

        # 1. 爆仓 (实际清算, 优先级最高)
        if rL <= liq_pct:
            return _exit(entry_spot, liq_price, hold, cfg, "爆仓",
                         d, is_liq=True)
        # 2. SL (主动止损, 距爆仓 buffer)
        if rL <= -sl:
            sl_price = entry_spot * (1 - sl / 100)
            return _exit(entry_spot, sl_price, hold, cfg, f"-{sl:.1f}% SL", d)
        # 3. TP
        if rH >= cfg.tp_pct:
            tp_price = entry_spot * (1 + cfg.tp_pct / 100)
            return _exit(entry_spot, tp_price, hold, cfg, f"+{cfg.tp_pct:.1f}% TP", d)
        # 4. Hold timeout
        if hold >= cfg.hold_max_days:
            return _exit(entry_spot, C, hold, cfg, f"{cfg.hold_max_days}d 时间出场", d)

    # daily 循环没触发退出 — 用 live 数据再检查今日盘中 + 实时 MTM
    if live_spot is not None and live_spot > 0:
        return _check_live(entry_spot, live_spot, live_high, live_low,
                             hold=hold, cfg=cfg, sl=sl, liq_pct=liq_pct,
                             liq_price=liq_price, today=today)
    # 没 live 数据兜底 — 用上日 close MTM
    last_d = later[-1] if len(later) else entry_d
    mtm_close = float(ohlc.loc[last_d, "Close"]) if last_d in ohlc.index else entry_spot
    return _exit(entry_spot, mtm_close, hold, cfg, "持仓中 MTM (无 live)", last_d, closed=False)


def _check_live(entry: float, live_spot: float,
                  live_high: float, live_low: float,
                  hold: int, cfg: FuturesConfig, sl: float,
                  liq_pct: float, liq_price: float,
                  today: pd.Timestamp) -> dict:
    """v3.7.134: 用今日 live spot/high/low 检查 SL/TP 是否已触发 (intraday).
    若没触发, 返回 OPEN with live MTM.
    """
    # 用 live_high/low 检查; 若没传, 用 live_spot 当点估
    H = live_high if live_high and live_high > 0 else live_spot
    L = live_low if live_low and live_low > 0 else live_spot
    rL = (L / entry - 1) * 100
    rH = (H / entry - 1) * 100
    # 1. 爆仓
    if rL <= liq_pct:
        return _exit(entry, liq_price, hold + 1, cfg, "爆仓 (intraday live)",
                       today, is_liq=True)
    # 2. SL
    if rL <= -sl:
        sl_price = entry * (1 - sl / 100)
        return _exit(entry, sl_price, hold + 1, cfg, f"-{sl:.1f}% SL (intraday live)",
                       today)
    # 3. TP
    if rH >= cfg.tp_pct:
        tp_price = entry * (1 + cfg.tp_pct / 100)
        return _exit(entry, tp_price, hold + 1, cfg,
                       f"+{cfg.tp_pct:.1f}% TP (intraday live)", today)
    # 4. OPEN — 用 live_spot 算 MTM
    return _exit(entry, live_spot, hold, cfg, "持仓中 (live MTM)",
                   today, closed=False)


def _exit(entry: float, exit_price: float, hold: int,
            cfg: FuturesConfig, reason: str, exit_d, is_liq: bool = False,
            closed: bool = True) -> dict:
    """统一退出包装 (含 funding + fee 净 PnL).

    爆仓: ROI on margin = -100% (保证金归零, 含强平滑点 + 保险金).
    """
    spot_pct = (exit_price / entry - 1) * 100
    if is_liq:
        # 爆仓: 保证金全损, ROI = -100%
        lev_pct = -100.0
        net_lev = -100.0
        funding_cost_pct = 0.0
        fee_pct = 0.0
        return {
            "closed": True, "exit_date": pd.Timestamp(exit_d),
            "entry_price": entry, "exit_price": exit_price,
            "hold_days": hold,
            "ret_spot_pct": spot_pct,
            "ret_levered_pct": -100.0,
            "net_pnl_pct": -100.0,
            "leverage": cfg.leverage,
            "liq_price": exit_price,
            "effective_sl_pct": effective_sl_pct(cfg),
            "reason": "爆仓 (保证金归零)",
            "is_liquidation": True,
            "pnl_pct": -100.0,
        }
    lev_pct = spot_pct * cfg.leverage
    # 资金费 (long 持仓时若 funding > 0 付费)
    n_funding = max(0, int(hold * 24 / 8))  # 每 8h 一次
    funding_cost_pct = cfg.funding_rate_8h * n_funding * 100  # spot %
    # 双边 taker fee (entry + exit)
    fee_pct = cfg.taker_fee * 2 * 100  # 在 spot % scale (但乘 lev 才是 ROI)
    # ROI on margin (实际收益率)
    net_spot = spot_pct - funding_cost_pct - fee_pct / cfg.leverage
    net_lev = net_spot * cfg.leverage
    return {
        "closed": closed,
        "exit_date": pd.Timestamp(exit_d),
        "entry_price": entry,
        "exit_price": exit_price,
        "hold_days": hold,
        "ret_spot_pct": spot_pct,
        "ret_levered_pct": lev_pct,                # 不扣费
        "net_pnl_pct": net_lev,                    # 扣费净 ROI
        "funding_cost_pct": funding_cost_pct * cfg.leverage,
        "fee_pct": fee_pct,
        "leverage": cfg.leverage,
        "liq_price": entry * (1 + liquidation_distance_pct(cfg, "long") / 100),
        "effective_sl_pct": effective_sl_pct(cfg),
        "reason": reason,
        "is_liquidation": is_liq,
        "pnl_pct": spot_pct,                       # 兼容 old API
    }


# ── 配置预设 (用户常见 leverage 选择) ──
LEVERAGE_PRESETS = {
    "Conservative_5x":  FuturesConfig(leverage=5,   tp_pct=8, sl_pct=5,  hold_max_days=15),
    "Moderate_10x":     FuturesConfig(leverage=10,  tp_pct=8, sl_pct=5,  hold_max_days=15),
    "Binance_20x":      FuturesConfig(leverage=20,  tp_pct=8, sl_pct=5,  hold_max_days=15),  # default
    "Aggressive_50x":   FuturesConfig(leverage=50,  tp_pct=5, sl_pct=1.0, hold_max_days=10),
    "Extreme_100x":     FuturesConfig(leverage=100, tp_pct=3, sl_pct=0.5, hold_max_days=5),
}


def get_preset(name: str) -> FuturesConfig:
    return LEVERAGE_PRESETS.get(name, LEVERAGE_PRESETS["Binance_20x"])
