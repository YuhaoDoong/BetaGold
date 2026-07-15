"""
改进模型集合

在 Phase 3 基线基础上，尝试多种改进方向，提升预测准确率。

模型列表:
1. CICC 4-Factor  — 中金黄金定价模型 (4因子线性)
2. CICC 3-Factor  — 中金2025版 (3因子, 去掉实际利率)
3. Ridge-Full     — Phase 3 基线 Ridge (116特征)
4. Ridge-Selected — LASSO 特征选择后的 Ridge (top特征)
5. XGB-Full       — Phase 3 基线 XGBoost
6. XGB-Tuned      — 调参后的 XGBoost (更保守防过拟合)
7. GARCH          — GARCH(1,1) 波动率预测 (仅 rv 目标)
8. Ridge-Regime   — Regime条件下的 Ridge (分高低波环境训练)
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. CICC 中金模型 (线性)
# ============================================================

class CICCModel:
    """
    中金公司黄金定价模型 — 线性回归

    4-Factor (2024版): real_yield_10y, tw_usd, cb_global_12m_rolling, federal_debt
    3-Factor (2025版): tw_usd, cb_global_12m_rolling, federal_debt (去掉实际利率)
    """

    FACTORS_4 = ["real_yield_10y", "tw_usd", "cb_global_12m_rolling", "federal_debt"]
    FACTORS_3 = ["tw_usd", "cb_global_12m_rolling", "federal_debt"]

    def __init__(self, n_factors: int = 4, alpha: float = 1.0):
        self.n_factors = n_factors
        self.factors = self.FACTORS_4 if n_factors == 4 else self.FACTORS_3
        self.name = f"CICC_{n_factors}F"
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cols = [c for c in self.factors if c in X.columns]
        if len(cols) < len(self.factors):
            missing = set(self.factors) - set(cols)
            logger.warning(f"  {self.name}: 缺少特征 {missing}")
        self._cols = cols
        X_sub = X[cols].ffill().fillna(0)
        self.model.fit(X_sub.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self._cols].ffill().fillna(0)
        return self.model.predict(X_sub.values)


# ============================================================
# 2. Ridge-Selected (LASSO 特征选择)
# ============================================================

class RidgeSelected:
    """
    先用 LASSO 选特征, 再用 Ridge 预测。
    减少噪声特征, 提升信号稳定性。
    """

    def __init__(self, lasso_alpha: float = 0.001, ridge_alpha: float = 1.0,
                 min_features: int = 15, max_features: int = 40):
        self.lasso_alpha = lasso_alpha
        self.ridge_alpha = ridge_alpha
        self.min_features = min_features
        self.max_features = max_features
        self.name = "Ridge-Sel"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Step 1: LASSO 选特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)

        lasso = Lasso(alpha=self.lasso_alpha, max_iter=5000)
        lasso.fit(X_scaled, y.values)

        # 按系数绝对值排序
        coef_abs = np.abs(lasso.coef_)
        ranked = np.argsort(-coef_abs)

        # 选取 non-zero 特征, 但不少于 min_features
        n_nonzero = np.sum(coef_abs > 0)
        n_select = max(self.min_features, min(n_nonzero, self.max_features))
        self._selected_idx = ranked[:n_select]
        self._selected_cols = [X.columns[i] for i in self._selected_idx]

        # Step 2: Ridge on selected features
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.ridge_alpha)),
        ])
        self.model.fit(X[self._selected_cols].values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._selected_cols].values)

    @property
    def selected_features(self) -> list:
        return self._selected_cols if hasattr(self, "_selected_cols") else []


# ============================================================
# 3. XGB-Tuned (更保守的 XGBoost)
# ============================================================

class XGBTuned:
    """
    调参后的 XGBoost — 更保守, 防过拟合
    - 更少树, 更浅, 更强正则
    """

    def __init__(self):
        self.name = "XGB-Tuned"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import xgboost as xgb
        self.model = xgb.XGBRegressor(
            n_estimators=150,       # 从300减到150
            max_depth=3,            # 从5减到3
            learning_rate=0.03,     # 从0.05减到0.03
            subsample=0.7,
            colsample_bytree=0.5,   # 从0.8减到0.5 (更强随机)
            min_child_weight=20,    # 从10增到20
            reg_alpha=0.5,          # 从0.1增到0.5
            reg_lambda=3.0,         # 从1.0增到3.0
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)


# ============================================================
# 4. GARCH (波动率专用)
# ============================================================

class GARCHModel:
    """
    GARCH(1,1) 波动率预测 — 仅用于 fwd_rv 目标。
    不依赖特征矩阵, 只用 GLD 收益率序列。
    """

    def __init__(self, p: int = 1, q: int = 1, horizon: int = 20):
        self.p = p
        self.q = q
        self.horizon = horizon
        self.name = f"GARCH({p},{q})"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X 必须包含 ret_1d 列。
        fit 只是存储最新的收益率序列用于 predict。
        """
        from arch import arch_model

        if "ret_1d" in X.columns:
            self._returns = X["ret_1d"].dropna() * 100  # arch 需要百分比
        else:
            logger.warning("  GARCH: 缺少 ret_1d, 使用 y 近似")
            self._returns = y * 100

        # 拟合 GARCH 模型
        self._am = arch_model(self._returns, vol="Garch",
                              p=self.p, q=self.q, mean="Zero")
        self._fit = self._am.fit(disp="off", show_warning=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        对每个测试日, 用 expanding 数据拟合 GARCH 并做 h-step 预测。
        为了效率, 每 21 天才重新拟合一次。
        """
        from arch import arch_model

        predictions = []
        test_dates = X.index

        # 获取完整收益率序列 (train + test)
        if "ret_1d" in X.columns:
            all_returns = pd.concat([
                self._returns,
                X["ret_1d"].dropna() * 100
            ]).sort_index()
            all_returns = all_returns[~all_returns.index.duplicated(keep="last")]
        else:
            all_returns = self._returns

        last_fit = None
        last_fit_idx = -999

        for i, date in enumerate(test_dates):
            # 每 21 天重新拟合
            if i - last_fit_idx >= 21 or last_fit is None:
                available = all_returns[all_returns.index <= date]
                if len(available) < 252:
                    predictions.append(np.nan)
                    continue
                try:
                    am = arch_model(available, vol="Garch",
                                    p=self.p, q=self.q, mean="Zero")
                    last_fit = am.fit(disp="off", show_warning=False)
                    last_fit_idx = i
                except Exception:
                    predictions.append(np.nan)
                    continue

            # h-step ahead 波动率预测
            try:
                fcast = last_fit.forecast(horizon=self.horizon)
                # 年化: sqrt(252) * sqrt(mean daily variance) / 100
                mean_var = fcast.variance.iloc[-1].mean()
                annual_vol = np.sqrt(mean_var) * np.sqrt(252) / 100
                predictions.append(annual_vol)
            except Exception:
                predictions.append(np.nan)

        return np.array(predictions)


# ============================================================
# 5. Ridge-Regime (Regime 条件模型)
# ============================================================

class RidgeRegime:
    """
    Regime 条件 Ridge — 按市场环境分别训练。

    使用 GVZ 分位数区分高波/低波环境, 对不同环境训练不同模型。
    测试时根据当前 regime 选择对应模型预测。
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.name = "Ridge-Regime"
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 判断 regime: 用 gvz_pctile_252d (如可用) 或 gvz_high
        if "gvz_high" in X.columns:
            regime = X["gvz_high"].fillna(0).astype(int)
        elif "gvz_pctile_252d" in X.columns:
            regime = (X["gvz_pctile_252d"] > 0.67).astype(int)
        else:
            # 无 regime 信息, 退化为普通 Ridge
            regime = pd.Series(0, index=X.index)

        self._regime_col = "gvz_high" if "gvz_high" in X.columns else "gvz_pctile_252d"

        # 对每种 regime 训练独立模型
        for r in regime.unique():
            mask = regime == r
            if mask.sum() < 100:
                continue
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ])
            model.fit(X[mask].values, y[mask].values)
            self.models[r] = model

        # 如果某 regime 样本太少, 用全量模型兜底
        self._fallback = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._fallback.fit(X.values, y.values)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "gvz_high" in X.columns:
            regime = X["gvz_high"].fillna(0).astype(int)
        elif "gvz_pctile_252d" in X.columns:
            regime = (X["gvz_pctile_252d"] > 0.67).astype(int)
        else:
            regime = pd.Series(0, index=X.index)

        predictions = np.zeros(len(X))
        for r in regime.unique():
            mask = regime == r
            if r in self.models:
                predictions[mask] = self.models[r].predict(X[mask].values)
            else:
                predictions[mask] = self._fallback.predict(X[mask].values)

        return predictions


# ============================================================
# 6. Ensemble (简单平均)
# ============================================================

class SimpleEnsemble:
    """
    简单集成 — 多模型预测的平均值。
    """

    def __init__(self, models: list):
        self.models = models
        names = [m.name for m in models]
        self.name = f"Ensemble({'+'.join(names)})"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for m in self.models:
            p = m.predict(X)
            preds.append(p)
        return np.nanmean(preds, axis=0)
