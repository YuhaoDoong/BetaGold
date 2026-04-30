"""波动率技术指标 — 用于 STRADDLE/SHORT_VOL 信号 (v3.7.44).

设计目标 (用户反馈):
  波动率交易主要应该看波动率本身的技术指标, 大事件只是辅助因素.
  事件不该是 score 主导, 而是技术指标 70% + 事件 30%.

提供的指标 (按用途):
  1. BBW pctile        — Bollinger Band Width 分位 (squeeze 检测)
  2. ATR ratio (5/20)  — 短期/长期 ATR 比 (波动收缩/扩张)
  3. Donchian width    — N 日最高最低范围 (range 收缩)
  4. RV %tile          — Realized Volatility 自身分位 (现有, 不重复)
  5. RV momentum       — RV 自身方向 (5d 变化, +↑ vs -↓)
  6. IV-RV spread      — 隐含 vs 实际差 (need IV proxy: ATM straddle 反推)

输出口径: 全部用 pctile 或 ratio (相对自身历史), 跨资产可比.

用法 (集成到 detect_straddle_signal):
  从 score 维度看:
    BBW < 20 分位 (squeeze) → +3 (最强 vol breakout 预兆)
    ATR ratio < 0.7 → +2 (短期收缩)
    RV %tile < 0.30 → +1 (低 IV 利于做多 vol)
    + 事件加分 (FOMC≤7d +1, 不再 +3)
  阈值: score ≥ 4 触发
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def bb_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Bollinger Band Width = (upper - lower) / mid.

    BBW 越小 → 价格波动收缩 → 即将突破 (vol expansion 预兆).
    """
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ((ma + k*sd) - (ma - k*sd)) / ma


def bbw_pctile(close: pd.Series, n: int = 20,
                  lookback: int = 252) -> pd.Series:
    """BBW 自身 1 年滚动分位 (0=极度 squeeze, 1=极度扩张)."""
    bbw = bb_width(close, n)
    return bbw.rolling(lookback, min_periods=60).rank(pct=True)


def true_range(high: pd.Series, low: pd.Series,
                close: pd.Series) -> pd.Series:
    """TR = max(H-L, |H-C_prev|, |L-C_prev|)."""
    pc = close.shift(1)
    return pd.concat([
        high - low,
        (high - pc).abs(),
        (low - pc).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
         n: int = 14) -> pd.Series:
    """ATR = TR 的 n 日 EMA (Wilder's smoothing 简化为 SMA)."""
    return true_range(high, low, close).rolling(n).mean()


def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series,
                short: int = 5, long: int = 20) -> pd.Series:
    """短 ATR / 长 ATR. <1 收缩 (vol breakout 概率↑), >1 扩张."""
    return atr(high, low, close, short) / atr(high, low, close, long)


def donchian_width(high: pd.Series, low: pd.Series,
                     n: int = 20) -> pd.Series:
    """Donchian channel width / midpoint 比例.

    n 日最高 - 最低 = 范围. 收缩到分位极低 → 突破前夜.
    """
    hi = high.rolling(n).max()
    lo = low.rolling(n).min()
    mid = (hi + lo) / 2
    return (hi - lo) / mid


def donchian_pctile(high: pd.Series, low: pd.Series, n: int = 20,
                       lookback: int = 252) -> pd.Series:
    """Donchian width 自身 1 年滚动分位."""
    dw = donchian_width(high, low, n)
    return dw.rolling(lookback, min_periods=60).rank(pct=True)


def rv_momentum(rv: pd.Series, lag: int = 5) -> pd.Series:
    """RV 5 日变化率 ((rv_t - rv_{t-5}) / rv_{t-5}).

    +0.3 = RV 5 日上升 30% → 波动加速期; 利于做多 vol 后期捕获
    -0.3 = RV 收敛中 → squeeze 中, 利于做多 vol 前期布局
    """
    return rv.pct_change(lag)


def iv_rv_spread_proxy(rv: pd.Series, gvz: Optional[pd.Series] = None
                         ) -> pd.Series:
    """IV - RV 差 (proxy). gvz 是 GLD 的 IV 指数 (类似 VIX for SPY).

    若没 GVZ, 退化用 rv.rolling(5) - rv 作为粗略 'forward implied' proxy.
    spread > 0: IV 贵 (做空 vol favorable)
    spread < 0: IV 便宜 (做多 vol favorable)
    """
    if gvz is not None:
        common = rv.index.intersection(gvz.index)
        return (gvz.loc[common] - rv.loc[common])
    # fallback: 5d rolling mean RV 当 implied 估计
    return rv.rolling(5).mean() - rv


def long_vol_score(close: pd.Series, high: pd.Series, low: pd.Series,
                     rv: pd.Series, rv_pctile: pd.Series,
                     gvz: Optional[pd.Series] = None) -> pd.DataFrame:
    """聚合所有做多 vol 技术信号 → score (0-10).

    7 个技术信号 + 后续可加 IV 真实数据.
      BBW <0.20 分位        +3 (强 squeeze)
      BBW 0.20-0.40        +1
      ATR ratio < 0.7        +2
      RV %tile < 0.30        +2
      RV %tile 0.30-0.50    +1
      Donchian width <0.20  +2
      RV momentum < -0.20    +1 (RV 仍在收缩)
      IV-RV spread < 0      +1 (IV 便宜)
    """
    out = pd.DataFrame(index=close.index)
    out['bbw_pct'] = bbw_pctile(close)
    out['atr_r'] = atr_ratio(high, low, close, 5, 20)
    out['donchian_pct'] = donchian_pctile(high, low)
    out['rv_pctile'] = rv_pctile
    out['rv_mom5'] = rv_momentum(rv, 5)
    out['iv_rv'] = iv_rv_spread_proxy(rv, gvz)

    score = pd.Series(0.0, index=close.index)
    score = score.add(np.where(out['bbw_pct'] < 0.20, 3, 0), fill_value=0)
    score = score.add(np.where((out['bbw_pct'] >= 0.20) & (out['bbw_pct'] < 0.40),
                                  1, 0), fill_value=0)
    score = score.add(np.where(out['atr_r'] < 0.7, 2, 0), fill_value=0)
    score = score.add(np.where(out['rv_pctile'] < 0.30, 2, 0), fill_value=0)
    score = score.add(np.where((out['rv_pctile'] >= 0.30) & (out['rv_pctile'] < 0.50),
                                  1, 0), fill_value=0)
    score = score.add(np.where(out['donchian_pct'] < 0.20, 2, 0), fill_value=0)
    score = score.add(np.where(out['rv_mom5'] < -0.20, 1, 0), fill_value=0)
    score = score.add(np.where(out['iv_rv'] < 0, 1, 0), fill_value=0)

    out['long_vol_tech_score'] = score
    return out


def short_vol_score(close: pd.Series, high: pd.Series, low: pd.Series,
                      rv: pd.Series, rv_pctile: pd.Series,
                      gvz: Optional[pd.Series] = None) -> pd.DataFrame:
    """做空 vol (Iron Condor) 技术信号 score (0-10).

      RV %tile > 0.70        +3 (高位卖 vol)
      RV %tile 0.50-0.70    +1
      RV momentum > 0.20    +1 (峰值后回落确认时减分)
      RV momentum < -0.20   +2 (RV 拐头下降)
      BBW pct > 0.70        +2 (扩张中段, 即将均值回归)
      ATR ratio > 1.3        +1 (短期高于长期, 极值)
      IV-RV spread > 0      +2 (IV 显著高于实际, 卖溢价)
    """
    out = pd.DataFrame(index=close.index)
    out['bbw_pct'] = bbw_pctile(close)
    out['atr_r'] = atr_ratio(high, low, close, 5, 20)
    out['rv_pctile'] = rv_pctile
    out['rv_mom5'] = rv_momentum(rv, 5)
    out['iv_rv'] = iv_rv_spread_proxy(rv, gvz)

    score = pd.Series(0.0, index=close.index)
    score = score.add(np.where(out['rv_pctile'] > 0.70, 3, 0), fill_value=0)
    score = score.add(np.where((out['rv_pctile'] > 0.50) & (out['rv_pctile'] <= 0.70),
                                  1, 0), fill_value=0)
    score = score.add(np.where(out['rv_mom5'] < -0.20, 2, 0), fill_value=0)
    score = score.add(np.where(out['bbw_pct'] > 0.70, 2, 0), fill_value=0)
    score = score.add(np.where(out['atr_r'] > 1.3, 1, 0), fill_value=0)
    score = score.add(np.where(out['iv_rv'] > 0, 2, 0), fill_value=0)

    out['short_vol_tech_score'] = score
    return out
