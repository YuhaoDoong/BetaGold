"""
基线模型: Ridge 线性回归 + XGBoost

两个模型共享统一接口: fit(X_train, y_train) / predict(X_test)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RidgeBaseline:
    """Ridge 回归基线 (含标准化)"""

    def __init__(self, alpha: float = 1.0, task: str = "regression"):
        self.task = task
        if task == "regression":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
        else:
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(
                    C=1.0 / alpha, max_iter=1000, solver="lbfgs",
                )),
            ])
        self.name = "Ridge" if task == "regression" else "LogReg"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X.values)
        return None


class XGBBaseline:
    """XGBoost 基线"""

    def __init__(self, task: str = "regression", **kwargs):
        self.task = task
        self.name = "XGBoost"
        self._params = kwargs

    def _build_model(self, n_classes: int = None):
        import xgboost as xgb
        base_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        base_params.update(self._params)

        if self.task == "regression":
            return xgb.XGBRegressor(**base_params)
        else:
            if n_classes and n_classes > 2:
                base_params["objective"] = "multi:softprob"
                base_params["num_class"] = n_classes
            return xgb.XGBClassifier(**base_params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        n_classes = int(y.nunique()) if self.task == "classification" else None
        self.model = self._build_model(n_classes)
        self.model.fit(X.values, y.values)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X.values)
        return None

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)
        return imp.head(top_n)
