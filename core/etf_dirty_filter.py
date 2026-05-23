"""v3.7.222: ETF 1h 脏 tick 实时检测 — 用 GC=F (干净, 23h 真期货) 当 reference.

原理:
  正常情况 GC=F.Close / ETF.Close 比例稳定在 ~10.86 (浮动 <1%).
  yfinance ETF 偶发 pre/post-market 合成 bar, 出现 wick 异常 (Low/High 偏离真值).
  脏 bar 时 ratio 急剧失衡 (>2%), GC=F 同时间 bar 一切正常.
  → ratio divergence > threshold → 该 ETF bar 标脏.

用法 (盘中实时):
  cleaned_etf = clean_etf_using_futures(etf_1h, gc_1h)
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def detect_dirty_bars(etf_1h: pd.DataFrame,
                       gc_1h: pd.DataFrame,
                       ratio_window: int = 60,
                       threshold_pct: float = 2.0) -> dict:
    """对 ETF 1h 每根 bar 用 GC=F 校验, 返回脏 bar 索引 + 诊断.

    Args:
        etf_1h: GLD/SLV 1h OHLC (yfinance ETF)
        gc_1h:  GC=F/SI=F 1h OHLC (yfinance futures, 干净)
        ratio_window: rolling baseline ratio 窗口 (默认 60 bar ≈ 2.5 天)
        threshold_pct: ETF Low/High 偏离 GC=F-推算值的 % 阈值 (默认 2%)

    Returns:
        dict with keys:
            dirty_indices: list of pd.Timestamp (脏 bar 时间)
            diagnostic: DataFrame with per-bar ratio, synth_low/high, divergence%
    """
    if (etf_1h is None or len(etf_1h) == 0
          or gc_1h is None or len(gc_1h) == 0):
        return {"dirty_indices": [], "diagnostic": pd.DataFrame()}

    common = etf_1h.index.intersection(gc_1h.index)
    if not len(common):
        return {"dirty_indices": [], "diagnostic": pd.DataFrame()}

    etf = etf_1h.loc[common].copy()
    gc = gc_1h.loc[common].copy()

    # 主参考: Close-to-Close ratio (rolling median 抗 outlier)
    ratio = gc["Close"] / etf["Close"]
    baseline = ratio.rolling(ratio_window, min_periods=20).median()
    # 起步阶段 baseline NaN → 用 expanding median 兜底
    baseline = baseline.fillna(ratio.expanding(min_periods=5).median())

    # GC-推算 ETF 等价 H/L
    synth_low = gc["Low"] / baseline
    synth_high = gc["High"] / baseline
    synth_close = gc["Close"] / baseline

    # ETF 偏离推算值的 %
    # 脏 wick LOW: ETF.Low << synth_low → low_dev > 0
    low_dev_pct = (synth_low - etf["Low"]) / synth_low * 100
    # 脏 wick HIGH: ETF.High >> synth_high → high_dev > 0
    high_dev_pct = (etf["High"] - synth_high) / synth_high * 100
    # Close 偏离 (作参考, 不直接判脏 — close 略漂移正常)
    close_dev_pct = ((etf["Close"] - synth_close) / synth_close * 100).abs()

    dirty_low_mask = low_dev_pct > threshold_pct
    dirty_high_mask = high_dev_pct > threshold_pct
    dirty_mask = dirty_low_mask | dirty_high_mask

    diag = pd.DataFrame({
        "etf_close": etf["Close"], "gc_close": gc["Close"],
        "etf_low": etf["Low"], "etf_high": etf["High"],
        "ratio": ratio, "baseline": baseline,
        "synth_low": synth_low, "synth_high": synth_high,
        "low_dev_pct": low_dev_pct, "high_dev_pct": high_dev_pct,
        "close_dev_pct": close_dev_pct,
        "etf_volume": etf["Volume"] if "Volume" in etf.columns else 0,
        "dirty": dirty_mask,
        "dirty_reason": np.where(
            dirty_low_mask & dirty_high_mask, "low+high wick",
            np.where(dirty_low_mask, "low wick",
                np.where(dirty_high_mask, "high wick", ""))),
    }, index=common)

    return {
        "dirty_indices": diag.index[dirty_mask].tolist(),
        "diagnostic": diag,
    }


def clean_etf_using_futures(etf_1h: pd.DataFrame,
                              gc_1h: pd.DataFrame,
                              ratio_window: int = 60,
                              threshold_pct: float = 2.0) -> tuple[pd.DataFrame, int]:
    """返回 (清理后的 ETF 1h DataFrame, 丢掉的 bar 数).
    脏 bar 直接 drop — 信号计算时 _new_l/_new_h 自然只看干净 bar."""
    result = detect_dirty_bars(etf_1h, gc_1h, ratio_window, threshold_pct)
    dirty_idx = result["dirty_indices"]
    if not dirty_idx:
        return etf_1h, 0
    cleaned = etf_1h.drop(index=dirty_idx, errors="ignore")
    return cleaned, len(dirty_idx)
