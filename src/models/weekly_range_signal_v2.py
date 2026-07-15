"""
周频区间交易信号 V2 (Phase 4A Step 3 redesign)

核心改进:
1. 公允价格 = EMA(price, 20) + macro_shift (替代 CICC 4因子 Ridge)
2. 区间宽度 = realized_vol × sqrt(5) × multiplier, GVZ/momentum 调整
3. 每周五重算公允价格和区间 (周频锁定)
4. Regime × band_position 决定仓位

因子分配:
- 公允价格中枢: EMA(price) + macro shift (DXY变化, 实际利率变化)
- 区间宽度: realized_vol, GVZ, momentum strength
- 方向: Regime (Bull/Mixed/Bear)

用法:
    from src.models.weekly_range_signal_v2 import AdaptiveFairValue, WeeklyRangeSignalV2
"""

import numpy as np
import pandas as pd


class AdaptiveFairValue:
    """
    自适应公允价格模型。

    fair_value = EMA(price, span) × (1 + macro_shift)

    macro_shift 基于近期宏观因子变化:
    - DXY 下跌 → 金价应更高 → shift > 0
    - 实际利率下降 → 金价应更高 → shift > 0

    每周五锁定一次, 周内不变。
    """

    def __init__(self, ema_span: int = 20,
                 usd_sensitivity: float = 0.5,
                 rate_sensitivity: float = 1.0,
                 max_shift: float = 0.01,
                 factor_lookback: int = 5):
        """
        ema_span: EMA 周期 (天)
        usd_sensitivity: USD 1%变化对应 fair_value 的调整幅度 (%)
        rate_sensitivity: 实际利率 1%变化对应的调整幅度 (%)
        max_shift: 单因子最大调整量 (防止极端)
        factor_lookback: 因子变化的回看天数
        """
        self.ema_span = ema_span
        self.usd_sensitivity = usd_sensitivity
        self.rate_sensitivity = rate_sensitivity
        self.max_shift = max_shift
        self.factor_lookback = factor_lookback

    def compute(self, gld_close: pd.Series,
                tw_usd: pd.Series = None,
                real_yield_10y: pd.Series = None,
                rebalance_freq: str = "W-FRI") -> pd.DataFrame:
        """
        计算公允价格和区间。

        返回 DataFrame: fair_value, macro_shift
        """
        idx = gld_close.index

        # 1. EMA 中枢
        ema_center = gld_close.ewm(span=self.ema_span, min_periods=10).mean()

        # 2. Macro shift
        macro_shift = pd.Series(0.0, index=idx)

        if tw_usd is not None:
            tw_chg = tw_usd.pct_change(self.factor_lookback).reindex(idx).fillna(0)
            # DXY 下跌 → shift 为正 (金价应该更高)
            usd_shift = (-tw_chg * self.usd_sensitivity).clip(
                -self.max_shift, self.max_shift)
            macro_shift = macro_shift + usd_shift

        if real_yield_10y is not None:
            ry_chg = real_yield_10y.diff(self.factor_lookback).reindex(idx).fillna(0)
            # 实际利率下降 → shift 为正
            rate_shift = (-ry_chg * self.rate_sensitivity).clip(
                -self.max_shift, self.max_shift)
            macro_shift = macro_shift + rate_shift

        # 3. Fair value = EMA × (1 + shift)
        fair_value = ema_center * (1 + macro_shift)

        # 4. 周频锁定
        rebal_dates = fair_value.resample(rebalance_freq).last().dropna().index
        fv_weekly = fair_value.reindex(rebal_dates).ffill()
        fv_daily = fv_weekly.reindex(idx).ffill()

        result = pd.DataFrame(index=idx)
        result["ema_center"] = ema_center
        result["macro_shift"] = macro_shift
        result["fair_value_raw"] = fair_value
        result["fair_value"] = fv_daily  # 周频锁定后

        return result


def compute_dynamic_band(gld_close: pd.Series,
                         fair_value: pd.Series,
                         vol_lookback: int = 20,
                         vol_multiplier: float = 2.0,
                         min_half_width: float = 0.015,
                         max_half_width: float = 0.08,
                         gvz: pd.Series = None,
                         ret_20d: pd.Series = None,
                         rebalance_freq: str = "W-FRI") -> pd.DataFrame:
    """
    动态区间宽度, 基于波动率 + GVZ + 动量。

    区间宽度因子 (影响 RANGE, 不影响中枢):
    - realized_vol: 基础宽度
    - GVZ 分位数: 隐含波动率偏高 → 区间加宽
    - 动量强度: 趋势强 → 区间加宽 (降低误触发)
    """
    idx = gld_close.index

    # 基础: realized_vol × sqrt(5) × multiplier
    daily_ret = gld_close.pct_change()
    rv = daily_ret.rolling(vol_lookback, min_periods=10).std()
    half_width = rv * np.sqrt(5) * vol_multiplier

    # GVZ 调整
    if gvz is not None and gvz.dropna().shape[0] > 60:
        gvz_aligned = gvz.reindex(idx)
        gvz_pctile = gvz_aligned.rolling(252, min_periods=60).apply(
            lambda x: (x.iloc[-1] >= x).mean() if len(x) > 0 else 0.5
        )
        gvz_adj = 1.0 + (gvz_pctile.fillna(0.5) - 0.5) * 0.5
        half_width = half_width * gvz_adj

    # 动量调整
    if ret_20d is not None:
        mom_strength = ret_20d.reindex(idx).abs() / 0.05
        mom_adj = 1.0 + np.clip(mom_strength, 0, 1) * 0.2
        half_width = half_width * mom_adj

    half_width = half_width.clip(min_half_width, max_half_width)

    # 周频锁定
    rebal_dates = half_width.resample(rebalance_freq).last().dropna().index
    hw_weekly = half_width.reindex(rebal_dates).ffill()
    hw_daily = hw_weekly.reindex(idx).ffill()

    # 区间
    upper = fair_value * (1 + hw_daily)
    lower = fair_value * (1 - hw_daily)
    band_range = (upper - lower).replace(0, np.nan)
    bp = (gld_close - lower) / band_range

    band = pd.DataFrame(index=idx)
    band["half_width"] = hw_daily
    band["upper"] = upper
    band["lower"] = lower
    band["band_position"] = bp

    return band


class WeeklyRangeSignalV2:
    """
    周频区间交易信号 V2。

    策略逻辑:
    - Regime 决定方向 (Bull/Mixed/Bear)
    - Band position 决定仓位大小
    - 每周五重算公允价格和区间

    Bull regime:
      bp < buy_zone → 满仓 (1.0)
      bp > sell_zone → 减仓至 bull_min (不清仓)
      中间 → 维持或默认 bull_default

    Mixed regime:
      bp < buy_zone → 加仓至 mixed_buy
      bp > sell_zone → 清仓 (0.0)
      中间 → 维持或默认 mixed_default

    Bear regime:
      仓位 = 0.0
    """

    def __init__(self,
                 # EMA 参数
                 ema_span: int = 20,
                 # Macro shift 参数
                 usd_sensitivity: float = 0.5,
                 rate_sensitivity: float = 1.0,
                 max_shift: float = 0.01,
                 factor_lookback: int = 5,
                 # Band 参数
                 vol_multiplier: float = 2.0,
                 min_half_width: float = 0.015,
                 max_half_width: float = 0.08,
                 # Position 参数
                 buy_zone: float = 0.20,
                 sell_zone: float = 0.85,
                 bull_min: float = 0.5,
                 bull_default: float = 0.7,
                 mixed_buy: float = 0.5,
                 mixed_default: float = 0.2):
        self.fv_model = AdaptiveFairValue(
            ema_span=ema_span,
            usd_sensitivity=usd_sensitivity,
            rate_sensitivity=rate_sensitivity,
            max_shift=max_shift,
            factor_lookback=factor_lookback,
        )
        self.vol_multiplier = vol_multiplier
        self.min_half_width = min_half_width
        self.max_half_width = max_half_width
        self.buy_zone = buy_zone
        self.sell_zone = sell_zone
        self.bull_min = bull_min
        self.bull_default = bull_default
        self.mixed_buy = mixed_buy
        self.mixed_default = mixed_default

    def generate(self,
                 regime: pd.Series,
                 gld_close: pd.Series,
                 tw_usd: pd.Series = None,
                 real_yield_10y: pd.Series = None,
                 gvz: pd.Series = None,
                 ret_20d: pd.Series = None,
                 rebalance_freq: str = "W-FRI") -> pd.DataFrame:
        """
        生成完整的交易信号。

        返回 DataFrame: fair_value, upper, lower, band_position,
                       regime, position, gld_close
        """
        idx = regime.index

        # 1. 公允价格
        fv_df = self.fv_model.compute(
            gld_close.reindex(idx),
            tw_usd=tw_usd,
            real_yield_10y=real_yield_10y,
            rebalance_freq=rebalance_freq,
        )

        # 2. 动态区间
        band_df = compute_dynamic_band(
            gld_close.reindex(idx),
            fv_df["fair_value"],
            vol_multiplier=self.vol_multiplier,
            min_half_width=self.min_half_width,
            max_half_width=self.max_half_width,
            gvz=gvz,
            ret_20d=ret_20d,
            rebalance_freq=rebalance_freq,
        )

        # 3. 合并结果
        result = pd.DataFrame(index=idx)
        result["gld_close"] = gld_close.reindex(idx)
        result["fair_value"] = fv_df["fair_value"]
        result["macro_shift"] = fv_df["macro_shift"]
        result["upper"] = band_df["upper"]
        result["lower"] = band_df["lower"]
        result["half_width"] = band_df["half_width"]
        result["band_position"] = band_df["band_position"]
        result["regime"] = regime

        # 4. 生成仓位
        bp = band_df["band_position"].values
        reg = regime.values
        positions = self._compute_positions(bp, reg)
        result["position"] = positions

        return result

    def _compute_positions(self, bp, reg):
        """根据 Regime × Band Position 计算仓位。"""
        n = len(bp)
        positions = np.zeros(n)
        prev = 0.0

        for i in range(n):
            b, r = bp[i], reg[i]
            if np.isnan(b) if isinstance(b, float) else False:
                positions[i] = prev
                continue

            if r == "Bull":
                if b <= self.buy_zone:
                    positions[i] = 1.0
                elif b >= self.sell_zone:
                    positions[i] = self.bull_min
                else:
                    # 中间区域: 维持, 但不低于 bull_min
                    positions[i] = max(prev, self.bull_default)
            elif r == "Mixed":
                if b <= self.buy_zone:
                    positions[i] = self.mixed_buy
                elif b >= self.sell_zone:
                    positions[i] = 0.0
                else:
                    positions[i] = min(prev, self.mixed_default) \
                        if prev > self.mixed_default else prev
            else:  # Bear
                positions[i] = 0.0

            prev = positions[i]

        return positions
