"""
公允价值区间模型 (Fair Value Band)

从"预测具体涨幅"转向"预测未来收益的分布区间"。

核心思路:
- 预测未来 N 天收益率的条件分位数 (q10/q25/q50/q75/q90)
- 结合当前价格，转换为未来价格区间
- 例如: 当前 GLD=$470, 20d q10=-3%, q90=+2% → 区间 [$455.9, $479.4]

模型列表:
1. CICC-Quantile    — 中金4因子 + 分位数回归
2. Ridge-Quantile   — 全特征分位数回归
3. XGB-Quantile     — XGBoost quantile regression
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

CICC_FACTORS = ["real_yield_10y", "tw_usd", "cb_global_12m_rolling", "federal_debt"]


class QuantileBandModel:
    """
    分位数区间模型基类。

    对每个分位数分别训练回归模型，预测未来 N 天收益率的条件分位数。
    推理时结合当前价格转为价格区间。
    """

    def __init__(self, name: str, quantiles: list = None):
        self.name = name
        self.quantiles = quantiles or QUANTILES
        self.models = {}
        self._scaler = None
        self._feature_cols = None

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """子类重写以选择特征子集"""
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: 特征矩阵
        y: 未来 N 天收益率 (fwd_ret_Nd)
        """
        X_sub = self._get_features(X)
        self._feature_cols = X_sub.columns.tolist()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_sub.values)

        for q in self.quantiles:
            model = self._build_model(q)
            model.fit(X_scaled, y.values)
            self.models[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        返回 DataFrame，列 q10~q90 为收益率分位数预测。
        """
        X_sub = X[self._feature_cols]
        X_scaled = self._scaler.transform(X_sub.values)

        results = {}
        for q in self.quantiles:
            pred = self.models[q].predict(X_scaled)
            results[f"q{int(q*100)}"] = pred

        return pd.DataFrame(results, index=X.index)

    def predict_price_band(self, X: pd.DataFrame,
                           current_price: pd.Series) -> pd.DataFrame:
        """
        返回价格区间: current_price * (1 + return_quantile)
        """
        ret_band = self.predict(X)
        price_band = pd.DataFrame(index=X.index)
        for col in ret_band.columns:
            price_band[col] = current_price * (1 + ret_band[col])
        return price_band

    def _build_model(self, q: float):
        raise NotImplementedError


class CICCQuantile(QuantileBandModel):
    """
    中金4因子 + 分位数回归。
    只用宏观因子，区间反映宏观条件下的合理价格范围。
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__(name="CICC-Quantile")
        self.alpha = alpha

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in CICC_FACTORS if c in X.columns]
        if len(cols) < len(CICC_FACTORS):
            missing = set(CICC_FACTORS) - set(cols)
            logger.warning(f"  {self.name}: 缺少特征 {missing}")
        return X[cols].ffill().fillna(0)

    def _build_model(self, q: float):
        return QuantileRegressor(
            quantile=q, alpha=self.alpha,
            solver="highs", fit_intercept=True,
        )


class RidgeQuantile(QuantileBandModel):
    """
    全特征分位数回归。
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__(name="Ridge-Quantile")
        self.alpha = alpha

    def _build_model(self, q: float):
        return QuantileRegressor(
            quantile=q, alpha=self.alpha,
            solver="highs", fit_intercept=True,
        )


class XGBQuantile(QuantileBandModel):
    """
    XGBoost quantile regression。
    """

    def __init__(self):
        super().__init__(name="XGB-Quantile")

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _build_model(self, q: float):
        import xgboost as xgb
        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=20,
            reg_alpha=0.5,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """XGBoost 不需要 StandardScaler"""
        X_sub = self._get_features(X)
        self._feature_cols = X_sub.columns.tolist()
        self._scaler = None

        for q in self.quantiles:
            model = self._build_model(q)
            model.fit(X_sub.values, y.values)
            self.models[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_sub = X[self._feature_cols]

        results = {}
        for q in self.quantiles:
            pred = self.models[q].predict(X_sub.values)
            results[f"q{int(q*100)}"] = pred

        return pd.DataFrame(results, index=X.index)
