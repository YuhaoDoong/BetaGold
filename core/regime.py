"""Regime 分类器 — 7因子规则打分, 非ML.

判断市场处于 Bull / Non-Bull 状态.
"""

import numpy as np
import pandas as pd


class RegimeClassifier:
    """多因子打分 Regime 分类器."""

    FACTOR_WEIGHTS = {
        "price_momentum": 0.25,
        "fed_rate_direction": 0.20,
        "usd_trend": 0.15,
        "central_bank": 0.15,
        "risk_sentiment": 0.10,
        "inflation": 0.10,
        "real_yield_level": 0.05,
    }

    def __init__(self, bull_threshold=0.2, bear_threshold=-0.2,
                 smooth_window=60, min_hold_days=20):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.smooth_window = smooth_window
        self.min_hold_days = min_hold_days

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """返回含 regime_score, regime 列的 DataFrame."""
        scores = pd.DataFrame(index=features.index)

        # 价格动量
        if "ret_60d" in features.columns:
            scores["price_momentum"] = np.clip(
                features["ret_60d"] / 0.10, -1, 1)
        elif "ret_20d" in features.columns:
            scores["price_momentum"] = np.clip(
                features["ret_20d"] / 0.05, -1, 1)
        else:
            scores["price_momentum"] = 0.0

        # 利率方向
        if "fed_funds_rate_change_60d" in features.columns:
            scores["fed_rate_direction"] = np.clip(
                -features["fed_funds_rate_change_60d"] / 0.5, -1, 1)
        else:
            scores["fed_rate_direction"] = 0.0

        # 实际利率水平
        if "real_yield_10y_zscore" in features.columns:
            scores["real_yield_level"] = np.clip(
                -features["real_yield_10y_zscore"] / 2, -1, 1)
        elif "real_yield_10y" in features.columns:
            scores["real_yield_level"] = np.clip(
                -(features["real_yield_10y"] - 0.5) / 1.0, -1, 1)
        else:
            scores["real_yield_level"] = 0.0

        # 美元趋势
        if "tw_usd_ret_20d" in features.columns:
            scores["usd_trend"] = np.clip(
                -features["tw_usd_ret_20d"] / 0.02, -1, 1)
        elif "tw_usd_zscore" in features.columns:
            scores["usd_trend"] = np.clip(
                -features["tw_usd_zscore"] / 2, -1, 1)
        else:
            scores["usd_trend"] = 0.0

        # 央行购金
        if "cb_global_12m_rolling" in features.columns:
            scores["central_bank"] = np.clip(
                (features["cb_global_12m_rolling"] - 200) / 300, -1, 1)
        else:
            scores["central_bank"] = 0.0

        # 风险情绪
        if "gvz_pctile_252d" in features.columns:
            scores["risk_sentiment"] = np.clip(
                (features["gvz_pctile_252d"].fillna(0.5) - 0.5) / 0.3, -1, 1)
        else:
            scores["risk_sentiment"] = 0.0

        # 通胀
        if "cpi_yoy" in features.columns:
            scores["inflation"] = np.clip(
                (features["cpi_yoy"] - 0.02) / 0.02, -1, 1)
        elif "breakeven_10y" in features.columns:
            scores["inflation"] = np.clip(
                (features["breakeven_10y"] - 2.0) / 0.5, -1, 1)
        else:
            scores["inflation"] = 0.0

        # 加权汇总
        composite = pd.Series(0.0, index=features.index)
        for factor, weight in self.FACTOR_WEIGHTS.items():
            if factor in scores.columns:
                composite += scores[factor].fillna(0) * weight

        scores["regime_score_raw"] = composite
        smoothed = composite.ewm(
            span=self.smooth_window, min_periods=20).mean()
        scores["regime_score"] = smoothed

        raw_regime = pd.Series("Mixed", index=features.index)
        raw_regime[smoothed > self.bull_threshold] = "Bull"
        raw_regime[smoothed < self.bear_threshold] = "Bear"

        if self.min_hold_days > 1:
            regime = self._apply_min_hold(raw_regime)
        else:
            regime = raw_regime

        scores["regime"] = regime
        return scores

    def _apply_min_hold(self, raw_regime: pd.Series) -> pd.Series:
        """最小持有期过滤."""
        values = raw_regime.values.copy()
        n = len(values)
        current = values[0]
        i = 1
        while i < n:
            if values[i] != current:
                new_regime = values[i]
                j = i
                while j < n and values[j] == new_regime:
                    j += 1
                if j - i < self.min_hold_days:
                    values[i:j] = current
                    i = j
                else:
                    current = new_regime
                    i = j
            else:
                i += 1
        return pd.Series(values, index=raw_regime.index)
