"""
周频区间交易信号 (Phase 4A Step 3)

两种模式:
1. RangeEdge: 区间边缘交易 (逢低买入, 逢高卖出)
2. RegimeOverlay: Regime 为主, 区间做风控叠加

核心组件:
- WeeklyFairValue: CICC 4因子 + 误差修正 → 每周重算公允价格
- compute_half_width: 波动率 + GVZ + 动量 → 动态区间宽度
- band_position: 价格在区间中的相对位置 (0=下界, 1=上界)

用法:
    from src.models.weekly_range_signal import WeeklyFairValue, WeeklyRangeSignal
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CICC_FACTORS = ["real_yield_10y", "tw_usd", "cb_global_12m_rolling", "federal_debt"]


class WeeklyFairValue:
    """
    CICC 4因子公允价格模型 + 误差修正。

    corrected_fv = raw_fv × EMA(actual / raw_fv, span)
    """

    def __init__(self, alpha: float = 1.0, correction_span: int = 60):
        self.alpha = alpha
        self.correction_span = correction_span
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self._cols = None

    def fit(self, X: pd.DataFrame, log_price: pd.Series):
        cols = [c for c in CICC_FACTORS if c in X.columns]
        self._cols = cols
        X_sub = X[cols].ffill().fillna(0)
        self.model.fit(X_sub.values, log_price.values)
        return self

    def predict_raw(self, X: pd.DataFrame) -> pd.Series:
        X_sub = X[self._cols].ffill().fillna(0)
        log_pred = self.model.predict(X_sub.values)
        return pd.Series(np.exp(log_pred), index=X.index, name="raw_fair_value")

    def predict_corrected(self, X: pd.DataFrame,
                          actual_price: pd.Series) -> pd.Series:
        raw_fv = self.predict_raw(X)
        ratio = actual_price / raw_fv
        correction = ratio.ewm(span=self.correction_span, min_periods=20).mean()
        corrected = raw_fv * correction
        return corrected.rename("fair_value")


def compute_half_width(gld_close: pd.Series,
                       vol_lookback: int = 20,
                       vol_multiplier: float = 2.0,
                       min_half_width: float = 0.015,
                       max_half_width: float = 0.08,
                       gvz: pd.Series = None,
                       ret_20d: pd.Series = None) -> pd.Series:
    """
    区间半宽 = realized_vol × multiplier × sqrt(5), 加 GVZ/动量调整。
    """
    daily_ret = gld_close.pct_change()
    rv = daily_ret.rolling(vol_lookback, min_periods=10).std()
    weekly_vol = rv * np.sqrt(5)
    half_width = weekly_vol * vol_multiplier

    if gvz is not None and len(gvz.dropna()) > 60:
        gvz_pctile = gvz.rolling(252, min_periods=60).apply(
            lambda x: (x.iloc[-1] >= x).mean() if len(x) > 0 else 0.5
        )
        gvz_adj = 1.0 + (gvz_pctile.fillna(0.5) - 0.5) * 0.6
        half_width = half_width * gvz_adj

    if ret_20d is not None:
        momentum_strength = ret_20d.abs() / 0.05
        mom_adj = 1.0 + np.clip(momentum_strength, 0, 1) * 0.2
        half_width = half_width * mom_adj

    return half_width.clip(min_half_width, max_half_width).rename("half_width")


def compute_band(fair_value: pd.Series, half_width: pd.Series,
                 gld_close: pd.Series, idx: pd.DatetimeIndex,
                 rebalance_freq: str = "W-FRI") -> pd.DataFrame:
    """计算周频锁定的区间和 band_position。"""
    rebal_dates = fair_value.resample(rebalance_freq).last().dropna().index
    fv_weekly = fair_value.reindex(rebal_dates).ffill()
    hw_weekly = half_width.reindex(rebal_dates).ffill()

    fv_daily = fv_weekly.reindex(idx).ffill()
    hw_daily = hw_weekly.reindex(idx).ffill()

    band = pd.DataFrame(index=idx)
    band["fair_value"] = fv_daily
    band["half_width"] = hw_daily
    band["upper"] = fv_daily * (1 + hw_daily)
    band["lower"] = fv_daily * (1 - hw_daily)
    band_range = (band["upper"] - band["lower"]).replace(0, np.nan)
    band["band_position"] = (gld_close - band["lower"]) / band_range
    return band


class WeeklyRangeSignal:
    """
    周频区间交易信号。

    两种模式 (mode):

    "edge": 区间边缘交易
      Bull/Mixed: bp < buy_zone → 满仓多, bp > sell_zone → 清仓, 中间维持
      Bear: bp > sell_zone → 做空, bp < buy_zone → 平空, 中间维持

    "overlay": Regime 为主, 区间做风控
      Bull: 默认满仓, bp > reduce_zone → 减仓, bp < dip_zone → 加回满仓
      Mixed: 默认 mixed_base, bp < dip_zone → 加仓至 mixed_dip
      Bear: 默认空仓
    """

    def __init__(self, mode: str = "overlay",
                 vol_multiplier: float = 2.0,
                 # Edge 模式参数
                 buy_zone: float = 0.25,
                 sell_zone: float = 0.75,
                 mixed_position: float = 0.5,
                 # Overlay 模式参数
                 bull_base: float = 1.0,
                 bull_reduce: float = 0.5,
                 reduce_zone: float = 0.85,
                 dip_zone: float = 0.30,
                 mixed_base: float = 0.0,
                 mixed_dip: float = 0.5):
        self.mode = mode
        self.vol_multiplier = vol_multiplier
        self.buy_zone = buy_zone
        self.sell_zone = sell_zone
        self.mixed_position = mixed_position
        self.bull_base = bull_base
        self.bull_reduce = bull_reduce
        self.reduce_zone = reduce_zone
        self.dip_zone = dip_zone
        self.mixed_base = mixed_base
        self.mixed_dip = mixed_dip

    def generate(self,
                 regime: pd.Series,
                 gld_close: pd.Series,
                 fair_value: pd.Series,
                 gvz: pd.Series = None,
                 ret_20d: pd.Series = None,
                 rebalance_freq: str = "W-FRI") -> pd.DataFrame:
        idx = regime.index
        hw = compute_half_width(gld_close, vol_multiplier=self.vol_multiplier,
                                gvz=gvz, ret_20d=ret_20d)
        band = compute_band(fair_value, hw, gld_close, idx, rebalance_freq)

        result = band.copy()
        result["regime"] = regime
        result["gld_close"] = gld_close

        bp = band["band_position"].values
        reg = regime.values

        if self.mode == "edge":
            positions = self._edge_positions(bp, reg)
        else:
            positions = self._overlay_positions(bp, reg)

        result["position"] = positions
        return result

    def _edge_positions(self, bp, reg):
        """区间边缘交易: 逢低买入, 逢高卖出。"""
        n = len(bp)
        positions = np.zeros(n)
        prev = 0.0

        for i in range(n):
            b, r = bp[i], reg[i]
            if np.isnan(b):
                positions[i] = prev
                continue

            if r in ("Bull", "Mixed"):
                sz = 1.0 if r == "Bull" else self.mixed_position
                if b <= self.buy_zone:
                    positions[i] = sz
                elif b >= self.sell_zone:
                    positions[i] = 0.0
                else:
                    positions[i] = prev
            elif r == "Bear":
                if b >= self.sell_zone:
                    positions[i] = -1.0
                elif b <= self.buy_zone:
                    positions[i] = 0.0
                else:
                    positions[i] = prev
            prev = positions[i]

        return positions

    def _overlay_positions(self, bp, reg):
        """Regime 为主, 区间做风控叠加。"""
        n = len(bp)
        positions = np.zeros(n)
        prev = 0.0

        for i in range(n):
            b, r = bp[i], reg[i]
            if np.isnan(b):
                positions[i] = prev
                continue

            if r == "Bull":
                # 默认满仓, 过热减仓, 回调恢复
                if b >= self.reduce_zone:
                    positions[i] = self.bull_reduce
                elif b <= self.dip_zone:
                    positions[i] = self.bull_base
                else:
                    # 中间区域: 维持, 但不低于 bull_reduce
                    positions[i] = max(prev, self.bull_reduce)
            elif r == "Mixed":
                # 默认 base 仓位, 逢低加仓
                if b <= self.dip_zone:
                    positions[i] = self.mixed_dip
                elif b >= self.sell_zone:
                    positions[i] = 0.0
                else:
                    positions[i] = min(prev, self.mixed_base) if prev > self.mixed_base else prev
            else:  # Bear
                positions[i] = 0.0

            prev = positions[i]

        return positions
