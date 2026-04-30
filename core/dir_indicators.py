"""方向性技术指标 — 用于:
  (A) 区间收紧 (post-process LSTM 输出, 让 upper/lower 更准)
  (B) 盘中确认 (价格触 band 边后, 必须技术确认才入场)

核心指标:
  RSI(14)        — 动量超买/超卖
  MACD hist      — 动量加速/减速
  ATR ratio 5/20 — 波动收缩/扩张 (用于 A)
  Volume ratio   — 量价确认 (用于 B)
  MA20 vs MA50   — 趋势方向
"""
from typing import Tuple
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI Wilder smoothing 简化为 SMA."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def atr_ratio_5_20(high: pd.Series, low: pd.Series,
                      close: pd.Series) -> pd.Series:
    """短/长 ATR 比 (复用 vol_indicators.atr_ratio 但内嵌避免循环导入)."""
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                     axis=1).max(axis=1)
    return tr.rolling(5).mean() / tr.rolling(20).mean()


def volume_ratio_20(volume: pd.Series) -> pd.Series:
    """当前成交量 / 20 日均量."""
    return volume / volume.rolling(20).mean()


def ma_trend(close: pd.Series) -> pd.Series:
    """MA20 / MA50 — >1 上行趋势, <1 下行."""
    return close.rolling(20).mean() / close.rolling(50).mean()


# ─────────────────────────────────────────────────────
# (A) 区间收紧 — 调整 LSTM 区间
# ─────────────────────────────────────────────────────
def adjust_band(upper: pd.Series, lower: pd.Series,
                  high: pd.Series, low: pd.Series, close: pd.Series,
                  ) -> Tuple[pd.Series, pd.Series]:
    """根据 ATR 收缩/扩张, 调整 LSTM 输出区间.

    ATR 5/20 < 0.7 (强收缩) → 区间打 ×0.85 (变窄)
    ATR 5/20 > 1.3 (扩张顶) → 区间打 ×1.15 (变宽)
    其他 → 不调

    返回 (upper_adj, lower_adj). bp 由调用者重算.
    """
    ar = atr_ratio_5_20(high, low, close)
    # 中点不变, 半宽缩放
    mid = (upper + lower) / 2
    half = (upper - lower) / 2
    factor = pd.Series(1.0, index=ar.index)
    factor = factor.where(ar >= 0.7, 0.85)   # ATR<0.7 → 0.85
    factor = factor.where(ar <= 1.3, 1.15)   # ATR>1.3 → 1.15
    half_adj = half * factor
    return mid + half_adj, mid - half_adj


# ─────────────────────────────────────────────────────
# (B) 盘中确认 — 价格触 band 时, 必须技术齐心
# ─────────────────────────────────────────────────────
def directional_confirm(close: pd.Series, high: pd.Series, low: pd.Series,
                          volume: pd.Series, side: str = "BUY") -> pd.DataFrame:
    """方向性技术确认 — buy-the-dip 反转确认 (v3.7.46 修正).

    bp ≤ 0.30 触发是反转/mean-revert, 不是 trend follow.
    BUY 确认条件 (oversold reversal):
      RSI < 40           (超卖区, 反弹空间大)
      MACD hist 上拐     (今日 hist > 昨日 hist, 转向中)
      Volume > 1.2× MA20 (放量, capitulation 后可能反弹)
      Close < MA20 × 0.99 (确实在 MA 下方, 真低位)

    SELL CALL (反向, overbought reversal):
      RSI > 60, MACD hist 下拐, Volume↑, Close > MA20×1.01
    """
    out = pd.DataFrame(index=close.index)
    out['rsi'] = rsi(close)
    _, _, hist = macd(close)
    out['macd_hist'] = hist
    out['macd_hist_d1'] = hist - hist.shift(1)  # 拐点
    out['vol_ratio'] = volume_ratio_20(volume)
    ma20 = close.rolling(20).mean()
    out['close_vs_ma20'] = close / ma20

    if side.upper() in ("BUY", "BUY CALL", "SELL PUT", "BULL"):
        c1 = out['rsi'] < 40                       # oversold
        c2 = out['macd_hist_d1'] > 0               # 拐点上行
        c3 = out['vol_ratio'] > 1.2                # 放量
        c4 = out['close_vs_ma20'] < 0.99           # MA 下方
    else:
        c1 = out['rsi'] > 60
        c2 = out['macd_hist_d1'] < 0
        c3 = out['vol_ratio'] > 1.2
        c4 = out['close_vs_ma20'] > 1.01

    confirm_count = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
    out['confirm_count'] = confirm_count
    out['confirm_2of4'] = confirm_count >= 2
    out['confirm_3of4'] = confirm_count >= 3
    return out
