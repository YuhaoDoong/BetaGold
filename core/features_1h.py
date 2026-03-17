"""1h 特征工程 — 多粒度特征构建.

从 1h K线数据构建三层特征:
  - 1h 技术指标 (微观: 日内波动节奏)
  - 4h 聚合特征 (中观: 2-3天波段)
  - 日线 聚合特征 + Regime (宏观: 周度趋势/宏观环境)

输出: DataFrame, index=1h datetime, 每根K线一行.
"""

import numpy as np
import pandas as pd


# ── 技术指标计算 ──

def _ret(close, period):
    return close.pct_change(period)


def _sma(s, n):
    return s.rolling(n, min_periods=1).mean()


def _ema(s, n):
    return s.ewm(span=n, min_periods=1).mean()


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    sig_line = _ema(macd_line, signal)
    return macd_line - sig_line  # histogram


def _bb_position(close, n=20):
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=1).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _bb_width(close, n=20):
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=1).std()
    return (4 * std / sma).replace(0, np.nan)


def _stoch(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    k = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    d = _sma(k, d_period)
    return k, d


def _atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _rv(close, n=10):
    """Realized volatility (annualized to 5-day scale)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n, min_periods=3).std() * np.sqrt(5) * 100


# ── 1h 层特征 ──

def build_1h_features(df):
    """从 1h OHLCV 构建 1h 级别技术特征.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume],
            index = datetime (1h bars).

    Returns: DataFrame with 1h features, same index.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # 收益率
    for p in [1, 3, 7, 14, 35]:
        feat[f"ret_{p}h"] = _ret(c, p)

    # 均线位置 (close / sma - 1)
    for n in [7, 14, 35, 70]:
        feat[f"close_sma{n}_pct"] = c / _sma(c, n) - 1

    # 均线斜率
    sma14 = _sma(c, 14)
    sma35 = _sma(c, 35)
    feat["sma14_slope"] = sma14.pct_change(7)
    feat["sma35_slope"] = sma35.pct_change(7)

    # 均线对齐
    sma7 = _sma(c, 7)
    feat["ma_alignment"] = ((sma7 > sma14) & (sma14 > sma35)).astype(float)

    # RSI
    feat["rsi_14"] = _rsi(c, 14)

    # MACD histogram
    feat["macd_hist"] = _macd(c)

    # 布林带
    feat["bb_position"] = _bb_position(c, 20)
    feat["bb_width"] = _bb_width(c, 20)

    # KDJ
    feat["stoch_k"], feat["stoch_d"] = _stoch(h, l, c, 14, 3)

    # ATR
    atr = _atr(h, l, c, 14)
    feat["atr_pct"] = atr / c

    # 日内范围
    feat["range_pct"] = (h - l) / c

    # RV
    feat["rv_10h"] = _rv(c, 10)
    feat["rv_35h"] = _rv(c, 35)

    # 成交量
    vol_sma = _sma(v, 14)
    feat["vol_ratio"] = v / vol_sma.replace(0, np.nan)

    # 价格位置 (在近N根的high-low范围内的位置)
    for n in [14, 35]:
        h_max = h.rolling(n, min_periods=1).max()
        l_min = l.rolling(n, min_periods=1).min()
        rng = (h_max - l_min).replace(0, np.nan)
        feat[f"price_pos_{n}h"] = (c - l_min) / rng

    return feat


# ── 4h 聚合特征 ──

def build_4h_features(df_1h):
    """从 1h 数据 resample 成 4h, 计算中观特征, 再对齐回 1h index.

    Returns: DataFrame with 4h features, index = 1h datetime (forward-filled).
    """
    # Resample to 4h
    df_4h = df_1h.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])

    c, h, l = df_4h["Close"], df_4h["High"], df_4h["Low"]

    feat_4h = pd.DataFrame(index=df_4h.index)

    # 收益率
    for p in [1, 3, 7]:
        feat_4h[f"4h_ret_{p}"] = _ret(c, p)

    # 均线位置
    for n in [5, 10, 20]:
        feat_4h[f"4h_sma{n}_pct"] = c / _sma(c, n) - 1

    # RSI
    feat_4h["4h_rsi"] = _rsi(c, 14)

    # BB
    feat_4h["4h_bb_pos"] = _bb_position(c, 20)

    # ATR%
    feat_4h["4h_atr_pct"] = _atr(h, l, c, 14) / c

    # RV
    feat_4h["4h_rv"] = _rv(c, 10)

    # Stoch
    feat_4h["4h_stoch_k"], _ = _stoch(h, l, c, 14, 3)

    # 对齐回 1h (forward fill)
    feat_4h_aligned = feat_4h.reindex(df_1h.index).ffill()
    return feat_4h_aligned


# ── 日线聚合特征 ──

def build_daily_features(df_1h, regime_series=None, daily_rv_pctile=None):
    """从 1h 数据聚合日线特征, 再对齐回 1h index.

    Args:
        df_1h: 1h OHLCV
        regime_series: 日线 Regime (pd.Series, index=date, values=Bull/Bear/Mixed)
        daily_rv_pctile: 日线 RV percentile (pd.Series, index=date)

    Returns: DataFrame, index = 1h datetime.
    """
    df_d = df_1h.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])

    c, h, l = df_d["Close"], df_d["High"], df_d["Low"]

    feat_d = pd.DataFrame(index=df_d.index)

    # 收益率
    for p in [1, 5, 10, 20]:
        feat_d[f"d_ret_{p}d"] = _ret(c, p)

    # 均线位置
    for n in [5, 20, 60]:
        feat_d[f"d_sma{n}_pct"] = c / _sma(c, n) - 1

    # RSI
    feat_d["d_rsi"] = _rsi(c, 14)

    # ATR%
    feat_d["d_atr_pct"] = _atr(h, l, c, 14) / c

    # RV
    feat_d["d_rv_20d"] = _rv(c, 20)

    # Regime (从现有日线模型)
    if regime_series is not None:
        regime_map = {"Bull": 1.0, "Mixed": 0.0, "Bear": -1.0}
        feat_d["d_regime"] = regime_series.reindex(df_d.index.normalize())\
            .map(regime_map).fillna(0)

    # RV percentile (从现有日线模型)
    if daily_rv_pctile is not None:
        feat_d["d_rv_pctile"] = daily_rv_pctile.reindex(
            df_d.index.normalize()).fillna(0.5)

    # 对齐回 1h
    feat_d_aligned = feat_d.reindex(df_1h.index).ffill()
    return feat_d_aligned


# ── 跨市场特征 ──

def build_cross_market_features(gc_1h, gld_1h=None, dxy_1h=None,
                                 vix_1h=None, slv_1h=None, tlt_1h=None):
    """从多个 1h 数据构建跨市场特征.

    gc_1h 为主, 所有输入 align 到 gc_1h 的 index.
    Returns: DataFrame, index = gc_1h datetime.
    """
    feat = pd.DataFrame(index=gc_1h.index)
    gc_c = gc_1h["Close"]

    if gld_1h is not None:
        gld_c = gld_1h["Close"].reindex(gc_1h.index).ffill()
        ratio = gc_c / gld_c.replace(0, np.nan)
        feat["gc_gld_ratio"] = ratio
        feat["gc_gld_ratio_z"] = (ratio - ratio.rolling(70).mean()) / \
            ratio.rolling(70).std().replace(0, np.nan)

    if dxy_1h is not None:
        dxy_c = dxy_1h["Close"].reindex(gc_1h.index).ffill()
        feat["dxy_ret_7h"] = dxy_c.pct_change(7)
        feat["dxy_ret_35h"] = dxy_c.pct_change(35)
        feat["dxy_sma20_pct"] = dxy_c / _sma(dxy_c, 20) - 1

    if vix_1h is not None:
        vix_c = vix_1h["Close"].reindex(gc_1h.index).ffill()
        feat["vix_level"] = vix_c
        feat["vix_ret_7h"] = vix_c.pct_change(7)
        feat["vix_sma20_dev"] = (vix_c - _sma(vix_c, 20)) / \
            _sma(vix_c, 20).replace(0, np.nan)

    if slv_1h is not None:
        slv_c = slv_1h["Close"].reindex(gc_1h.index).ffill()
        gs_ratio = gc_c / slv_c.replace(0, np.nan)
        feat["gold_silver_ratio"] = gs_ratio
        feat["gold_silver_change"] = gs_ratio.pct_change(7)

    if tlt_1h is not None:
        tlt_c = tlt_1h["Close"].reindex(gc_1h.index).ffill()
        feat["tlt_ret_7h"] = tlt_c.pct_change(7)
        feat["gc_tlt_corr_35h"] = gc_c.pct_change().rolling(35).corr(
            tlt_c.pct_change())

    return feat


# ── 预测目标 ──

def build_targets(df_1h, horizons=(19, 95)):
    """构建多时间尺度预测目标.

    Args:
        df_1h: 1h OHLCV (GC=F: ~19根/天, 所以19≈1天, 95≈5天)
        horizons: tuple of forward periods in 1h bars

    Returns: DataFrame with target columns.
    """
    c, h, l = df_1h["Close"], df_1h["High"], df_1h["Low"]
    targets = pd.DataFrame(index=df_1h.index)

    for nh in horizons:
        # 未来 nh 根K线的最高价和最低价
        fwd_high = h.rolling(nh).max().shift(-nh)
        fwd_low = l.rolling(nh).min().shift(-nh)

        targets[f"fwd_{nh}h_upper_pct"] = (fwd_high / c - 1) * 100
        targets[f"fwd_{nh}h_lower_pct"] = (fwd_low / c - 1) * 100

    return targets


# ── 主入口: 构建完整数据集 ──

def build_dataset(gc_1h, gld_1h=None, dxy_1h=None, vix_1h=None,
                  slv_1h=None, tlt_1h=None,
                  regime_series=None, daily_rv_pctile=None,
                  horizons=(19, 95)):
    """构建完整的 1h 特征+目标数据集.

    以 GC=F (COMEX黄金期货) 为主信号源 — 全球24h交易,
    ~19根/天, 5天≈95根. 覆盖亚洲/伦敦/纽约全时段.

    GLD 等 ETF 作为跨市场参考特征.

    Args:
        gc_1h: GC=F 1h OHLCV (主数据源)
        gld_1h: GLD 1h OHLCV (跨市场特征)
        horizons: (19, 95) = (1天, 5天) on GC=F

    Returns: (features_df, targets_df), 已对齐, 包含NaN (由调用方处理).
    """
    # 1h 层 (GC=F)
    feat_1h = build_1h_features(gc_1h)

    # 4h 层 (GC=F)
    feat_4h = build_4h_features(gc_1h)

    # 日线层 (GC=F + Regime)
    feat_d = build_daily_features(gc_1h, regime_series, daily_rv_pctile)

    # 跨市场: GC=F 为主, GLD/DXY/VIX/SLV/TLT 为参考
    feat_cross = build_cross_market_features(
        gc_1h, gld_1h, dxy_1h, vix_1h, slv_1h, tlt_1h)

    # 合并
    features = pd.concat([feat_1h, feat_4h, feat_d, feat_cross], axis=1)

    # 目标 (GC=F 价格区间)
    targets = build_targets(gc_1h, horizons)

    return features, targets
