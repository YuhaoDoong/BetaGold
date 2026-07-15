"""
组合信号: Regime + 公允价值偏离度 → 仓位信号 (Phase 4A Step 3)

核心思路:
1. 用 CICC 4因子回归出"公允价格" (fair value)
2. 偏离度 = (实际价格 - 公允价格) / 公允价格
3. 用 rolling 分位数归一化偏离度 (vs 近1年), 得到 0~1 的相对位置
4. 结合 Regime 决定仓位:
   - Bull + 相对便宜 (分位数<30%) → 积极做多
   - Bull + 中性 → 轻仓持有
   - Bull + 相对贵 (分位数>80%) → 减仓
   - Non-Bull → 空仓或极轻仓
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CICC_FACTORS = ["real_yield_10y", "tw_usd", "cb_global_12m_rolling", "federal_debt"]


class FairValueModel:
    """CICC 4因子公允价格模型"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
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

    def predict_fair_value(self, X: pd.DataFrame) -> pd.Series:
        """返回公允价格 (原始价格空间)"""
        X_sub = X[self._cols].ffill().fillna(0)
        log_pred = self.model.predict(X_sub.values)
        return pd.Series(np.exp(log_pred), index=X.index, name="fair_value")


class CombinedSignal:
    """
    Regime + 公允价值偏离度 → 仓位信号
    """

    def __init__(self, lookback: int = 252,
                 cheap_threshold: float = 0.30,
                 expensive_threshold: float = 0.80):
        """
        lookback: 偏离度分位数的回看窗口 (天)
        cheap_threshold: 分位数低于此值 = 相对便宜
        expensive_threshold: 分位数高于此值 = 相对贵
        """
        self.lookback = lookback
        self.cheap_threshold = cheap_threshold
        self.expensive_threshold = expensive_threshold
        self.name = "CombinedSignal"

    def generate(self, regime: pd.Series,
                 gld_close: pd.Series,
                 fair_value: pd.Series) -> pd.DataFrame:
        """
        生成组合信号。

        regime: "Bull" / "Mixed" / "Bear"
        gld_close: GLD 收盘价
        fair_value: 模型预测的公允价格

        返回 DataFrame: position, deviation, dev_pctile, zone, regime
        """
        result = pd.DataFrame(index=regime.index)
        result["regime"] = regime
        result["gld_close"] = gld_close
        result["fair_value"] = fair_value

        # 偏离度
        deviation = (gld_close - fair_value) / fair_value
        result["deviation"] = deviation

        # Rolling 分位数 (vs 近 lookback 天)
        dev_pctile = deviation.rolling(
            self.lookback, min_periods=self.lookback // 2
        ).apply(lambda x: (x.iloc[-1] >= x).mean() if len(x) > 0 else 0.5)
        result["dev_pctile"] = dev_pctile

        # 区域划分
        zone = pd.Series("neutral", index=regime.index)
        zone[dev_pctile <= self.cheap_threshold] = "cheap"
        zone[dev_pctile >= self.expensive_threshold] = "expensive"
        result["zone"] = zone

        # 仓位映射
        is_bull = regime == "Bull"
        position = pd.Series(0.0, index=regime.index)

        # Bull regime
        position[is_bull & (zone == "cheap")] = 1.0        # 相对便宜, 满仓
        position[is_bull & (zone == "neutral")] = 0.5       # 中性, 半仓
        position[is_bull & (zone == "expensive")] = 0.0     # 相对贵, 减仓

        # Non-Bull regime
        position[~is_bull & (zone == "cheap")] = 0.3        # 逢低轻仓
        position[~is_bull & (zone == "neutral")] = 0.0      # 空仓
        position[~is_bull & (zone == "expensive")] = 0.0    # 空仓

        result["position"] = position
        return result
