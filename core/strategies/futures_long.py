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
    """期货策略参数 (paper_positions + backtest 共用).

    v3.7.137: 统一用 margin % 表示 TP/SL (与 BC/SP/IC 对齐, 跟"策略盈亏"挂钩).
              spot % 通过 leverage 换算: spot_pct = margin_pct / leverage
    """
    leverage: int = 20                   # 杠杆 (默认/B tier, Binance XAUUSDT 上限 20)
    maintenance_margin_rate: float = 0.005  # MM rate (Bracket 1 < $50k notional)

    # v3.7.204: per-tier leverage 配置 (None = 用 leverage 默认)
    # 高质量信号 (S/A 历史 100% WR) 上更高杠杆, 标准信号保守
    # 注: lev=15 爆距 ≈ 6.2% spot, GC=F 单日 wick 3-19 -5.7% / 3-23 -5.8% 接近爆仓
    #     用 v3.7.201 信号过滤后 Q1 残留 1 笔, lev=15 在 wick 期可能爆
    tier_s_leverage: int = None          # 例 15 (S 最优, 100% WR)
    tier_a_leverage: int = None          # 例 10
    tier_b_leverage: int = None          # 例 5

    # per-tier 仓位 fraction (None = 1.0 全仓; Kelly 推荐 S=1.0 A=0.8 B=0.5)
    tier_s_position_frac: float = 1.0
    tier_a_position_frac: float = 1.0
    tier_b_position_frac: float = 1.0

    # 主退出参数 — margin % (策略盈亏视角, 但客观 grid 最优 = 100% margin = 5% spot)
    # v3.7.138: 之前 sl_margin=50% 是"心理可控"非客观最优. exit_params_grid 显示
    #          sl=5% spot (= 100% margin @ 20×) 累计 +571% 是 grid 真最优.
    tp_margin_pct: float = 200.0         # +200% margin = +10% spot @ 20×
    sl_margin_pct: float = 100.0         # -100% margin = -5% spot @ 20× (客观最优)
    hold_max_days: int = 15

    # 爆仓感知 — 距爆仓价 buffer (默认关, 因 sl_margin 已远高于爆仓)
    auto_sl_from_liq: bool = False       # True=自动收紧到 (爆仓 - buffer) 之内
    liq_buffer_pct: float = 1.0          # 距爆仓 buffer (% spot)

    # v3.7.135: 早平 (持仓 N 天后, 利润 ≥ M% 即平 — 防被反向拖死回亏)
    # 实务: 持 3-5d 已 +5% 应当锁利, 不该贪等 +12% 回吐到 -70%
    early_tp_locks: tuple = (
        (3, 5.0),   # 持 3d, 利润 ≥ 5% → 平 (= +100% margin @ 20×)
        (7, 3.0),   # 持 7d, 利润 ≥ 3% → 平 (= +60% margin)
        (10, 1.0),  # 持 10d, 利润 ≥ 1% → 平 (= +20% margin)
    )

    # v3.7.135: 信号反转退出 (bp_high > 阈值时即使没到 TP 也平)
    signal_reversal_bp_high: float = 0.85  # 区间上沿即视反转
    signal_reversal_min_profit: float = 0.0  # 反转 + 至少 0% 利润才平

    # 费用 (v3.7.204: Binance USDM XAUUSDT 实测校准)
    # taker_fee: Binance USDM standard taker 0.05% (regular, no VIP/BNB)
    # funding_rate_8h: Binance XAUUSDT 33d 实测 mean=-0.00200% (long 净收 funding)
    #   旧 0.0001 (long 付 0.01%/8h) 是 spot 期货笼统估算, 严重高估成本
    #   实际 long 持仓: 年化 funding 收益 ~+2.2%, 不是 -10%+
    taker_fee: float = 0.0005            # 单边 0.05% (Binance USDM standard taker)
    funding_rate_8h: float = -0.00002    # -0.002%/8h (Binance XAUUSDT 实测 mean)


def liquidation_distance_pct(cfg: FuturesConfig, side: str = "long") -> float:
    """相对 entry 的爆仓价距离 (% spot, long 为负, short 为正)."""
    base = 1.0 / cfg.leverage - cfg.maintenance_margin_rate
    return -base * 100 if side == "long" else base * 100


def sl_spot_pct(cfg: FuturesConfig, side: str = "long") -> float:
    """实际止损 spot % (从 sl_margin_pct 换算)."""
    sl_spot = cfg.sl_margin_pct / cfg.leverage
    if cfg.auto_sl_from_liq:
        liq_dist = abs(liquidation_distance_pct(cfg, side))
        safe_sl = liq_dist - cfg.liq_buffer_pct
        sl_spot = min(sl_spot, max(0.5, safe_sl))
    return sl_spot


def tp_spot_pct(cfg: FuturesConfig) -> float:
    """实际止盈 spot % (从 tp_margin_pct 换算)."""
    return cfg.tp_margin_pct / cfg.leverage


# 兼容旧 API
def effective_sl_pct(cfg: FuturesConfig, side: str = "long") -> float:
    return sl_spot_pct(cfg, side)


def simulate_long_position(entry_d: pd.Timestamp,
                              entry_spot: float,
                              ohlc: pd.DataFrame,
                              today: pd.Timestamp,
                              cfg: FuturesConfig = None,
                              live_spot: float = None,
                              live_high: float = None,
                              live_low: float = None,
                              bp_high_series: pd.Series = None,
                              signal_tier: str = None) -> dict:
    """模拟期货多头持仓 — 4 层退出: 爆仓 / SL / TP / Timeout.

    v3.7.134: 期货 24h 可交易 (Binance XAUUSDT / GC=F COMEX 23h).
      live_spot: 当前实时价 (用于 MTM, 不再用上日 close)
      live_high/low: 今日盘中 high/low (用于检查 SL/TP 是否已被触发)
                     若 None, 用 live_spot 当 high=low (保守 — 仅 close 检测)
    v3.7.204: signal_tier (S/A/B) — 启用 per-tier leverage 覆盖.
    时间序: bar 内最坏先发 (保守).
    """
    if cfg is None: cfg = FuturesConfig()
    # v3.7.204: per-tier leverage 覆盖
    if signal_tier == "S" and cfg.tier_s_leverage is not None:
        import copy as _copy; cfg = _copy.copy(cfg); cfg.leverage = cfg.tier_s_leverage
    elif signal_tier == "A" and cfg.tier_a_leverage is not None:
        import copy as _copy; cfg = _copy.copy(cfg); cfg.leverage = cfg.tier_a_leverage
    elif signal_tier == "B" and cfg.tier_b_leverage is not None:
        import copy as _copy; cfg = _copy.copy(cfg); cfg.leverage = cfg.tier_b_leverage
    sl = sl_spot_pct(cfg)        # spot % SL (= sl_margin_pct / leverage)
    tp = tp_spot_pct(cfg)        # spot % TP (= tp_margin_pct / leverage)
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
        # 2. SL (主动止损 — margin % 视角)
        if rL <= -sl:
            sl_price = entry_spot * (1 - sl / 100)
            return _exit(entry_spot, sl_price, hold, cfg,
                          f"-{cfg.sl_margin_pct:.0f}% margin SL (spot -{sl:.1f}%)", d)
        # 3. TP (margin % 视角)
        if rH >= tp:
            tp_price = entry_spot * (1 + tp / 100)
            return _exit(entry_spot, tp_price, hold, cfg,
                          f"+{cfg.tp_margin_pct:.0f}% margin TP (spot +{tp:.1f}%)", d)
        # 3b. v3.7.135: 早平 (利润随持仓时间递减阈值, 防被反向拖死)
        rC = (C / entry_spot - 1) * 100
        for _hold_d, _min_profit in cfg.early_tp_locks:
            if hold >= _hold_d and rC >= _min_profit:
                return _exit(entry_spot, C, hold, cfg,
                              f"{_hold_d}d+{_min_profit:.0f}% 早平锁利", d)
        # 3c. v3.7.135: 信号反转 (bp_high > 0.85 + 已盈利 → 平)
        if bp_high_series is not None and pd.Timestamp(d) in bp_high_series.index:
            bph = float(bp_high_series.get(pd.Timestamp(d), 0))
            if bph > cfg.signal_reversal_bp_high \
               and rC >= cfg.signal_reversal_min_profit:
                return _exit(entry_spot, C, hold, cfg,
                              f"bp_high {bph:.2f} 反转 +{rC:.1f}% 平", d)
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
    # 2. SL (margin % 视角)
    if rL <= -sl:
        sl_price = entry * (1 - sl / 100)
        return _exit(entry, sl_price, hold + 1, cfg,
                       f"-{cfg.sl_margin_pct:.0f}% margin SL (live spot -{sl:.1f}%)",
                       today)
    # 3. TP (margin % 视角)
    tp = tp_spot_pct(cfg)
    if rH >= tp:
        tp_price = entry * (1 + tp / 100)
        return _exit(entry, tp_price, hold + 1, cfg,
                       f"+{cfg.tp_margin_pct:.0f}% margin TP (live spot +{tp:.1f}%)",
                       today)
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


# ── 配置预设 (各杠杆下的统一 margin %) ──
# v3.7.137: TP/SL 用 margin % (与 BC/SP 对齐). spot % = margin / leverage 自动换算.
LEVERAGE_PRESETS = {
    "Conservative_5x":  FuturesConfig(leverage=5,   tp_margin_pct=200, sl_margin_pct=50,  hold_max_days=15),
    "Moderate_10x":     FuturesConfig(leverage=10,  tp_margin_pct=200, sl_margin_pct=50,  hold_max_days=15),
    "Binance_20x":      FuturesConfig(leverage=20,  tp_margin_pct=200, sl_margin_pct=50,  hold_max_days=15),  # default
    "Aggressive_50x":   FuturesConfig(leverage=50,  tp_margin_pct=200, sl_margin_pct=50,  hold_max_days=10,
                                         auto_sl_from_liq=True),  # lev 高时强制爆仓 buffer
    "Extreme_100x":     FuturesConfig(leverage=100, tp_margin_pct=200, sl_margin_pct=50,  hold_max_days=5,
                                         auto_sl_from_liq=True),
}


def get_preset(name: str) -> FuturesConfig:
    return LEVERAGE_PRESETS.get(name, LEVERAGE_PRESETS["Binance_20x"])
