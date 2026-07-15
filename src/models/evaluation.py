"""
模型评估指标

回归: R², MAE, RMSE, IC (信息系数), 方向准确率
分类: Accuracy, AUC, Precision, Recall
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report,
)
from scipy.stats import spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """回归评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return {}

    # IC: Spearman 秩相关 (信息系数)
    ic, ic_pval = spearmanr(yt, yp)

    # 方向准确率 (预测符号正确)
    dir_acc = np.mean(np.sign(yt) == np.sign(yp))

    return {
        "R2": r2_score(yt, yp),
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": np.sqrt(mean_squared_error(yt, yp)),
        "IC": ic,
        "IC_pval": ic_pval,
        "Dir_Acc": dir_acc,
        "N": len(yt),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray = None) -> dict:
    """分类评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask].astype(int), y_pred[mask].astype(int)
    if len(yt) < 10:
        return {}

    n_classes = len(np.unique(yt))
    avg = "binary" if n_classes == 2 else "macro"

    result = {
        "Accuracy": accuracy_score(yt, yp),
        "Precision": precision_score(yt, yp, average=avg, zero_division=0),
        "Recall": recall_score(yt, yp, average=avg, zero_division=0),
        "N": len(yt),
    }

    # AUC (需要概率)
    if y_prob is not None:
        prob = y_prob[mask]
        try:
            if n_classes == 2:
                result["AUC"] = roc_auc_score(yt, prob[:, 1])
            else:
                result["AUC"] = roc_auc_score(
                    yt, prob, multi_class="ovr", average="macro"
                )
        except (ValueError, IndexError):
            pass

    return result


def summarize_fold_results(results: list[dict]) -> pd.DataFrame:
    """汇总多个 fold 的评估结果"""
    df = pd.DataFrame(results)
    summary = df.describe().loc[["mean", "std", "min", "max"]]
    return summary


# ======================================================================
# IC / ICIR 因子评估体系 (借鉴 Qlib 评估标准)
# ======================================================================

def rolling_ic(factor: pd.Series, forward_ret: pd.Series,
               window: int = 20, method: str = "spearman") -> pd.Series:
    """
    计算因子的 rolling IC (信息系数)。

    Parameters
    ----------
    factor : 因子值 (日频)
    forward_ret : 未来 N 日收益率 (与 factor 同 index)
    window : 滚动窗口天数
    method : "spearman" (Rank IC) 或 "pearson" (IC)

    Returns
    -------
    pd.Series: 滚动 IC 时序
    """
    df = pd.DataFrame({"factor": factor, "ret": forward_ret}).dropna()
    if method == "spearman":
        ic_series = df["factor"].rolling(window).corr(
            df["ret"].rolling(window).rank(pct=True)
        )
        # 更准确: 用 rank-rank 相关
        ic_series = (
            df["factor"].rank(pct=True)
            .rolling(window)
            .corr(df["ret"].rank(pct=True).rolling(window).apply(lambda x: x.iloc[-1], raw=False).shift(0))
        )
        # 简化: 逐窗口 spearman
        ic_vals = []
        idx = df.index
        fv = df["factor"].values
        rv = df["ret"].values
        for i in range(len(df)):
            if i < window - 1:
                ic_vals.append(np.nan)
            else:
                f_w = fv[i - window + 1: i + 1]
                r_w = rv[i - window + 1: i + 1]
                mask = ~(np.isnan(f_w) | np.isnan(r_w))
                if mask.sum() < 5:
                    ic_vals.append(np.nan)
                else:
                    ic_vals.append(spearmanr(f_w[mask], r_w[mask])[0])
        ic_series = pd.Series(ic_vals, index=idx, name="IC")
    else:
        from scipy.stats import pearsonr
        ic_vals = []
        idx = df.index
        fv = df["factor"].values
        rv = df["ret"].values
        for i in range(len(df)):
            if i < window - 1:
                ic_vals.append(np.nan)
            else:
                f_w = fv[i - window + 1: i + 1]
                r_w = rv[i - window + 1: i + 1]
                mask = ~(np.isnan(f_w) | np.isnan(r_w))
                if mask.sum() < 5:
                    ic_vals.append(np.nan)
                else:
                    ic_vals.append(pearsonr(f_w[mask], r_w[mask])[0])
        ic_series = pd.Series(ic_vals, index=idx, name="IC")

    return ic_series


def factor_ic_summary(factor: pd.Series, forward_ret: pd.Series,
                      window: int = 20) -> dict:
    """
    单因子 IC 摘要统计。

    Returns
    -------
    dict with keys: IC_mean, IC_std, ICIR, IC_pos_ratio, IC_abs_mean
    """
    ic = rolling_ic(factor, forward_ret, window=window, method="spearman")
    ic_clean = ic.dropna()
    if len(ic_clean) < 10:
        return {"IC_mean": np.nan, "IC_std": np.nan, "ICIR": np.nan,
                "IC_pos_ratio": np.nan, "IC_abs_mean": np.nan, "N": 0}

    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()
    icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

    return {
        "IC_mean": round(ic_mean, 4),
        "IC_std": round(ic_std, 4),
        "ICIR": round(icir, 4),
        "IC_pos_ratio": round((ic_clean > 0).mean(), 4),
        "IC_abs_mean": round(ic_clean.abs().mean(), 4),
        "N": len(ic_clean),
    }


def evaluate_all_factors(features_df: pd.DataFrame,
                         forward_ret: pd.Series,
                         ic_window: int = 20,
                         min_non_null: float = 0.5) -> pd.DataFrame:
    """
    对所有因子做 IC/ICIR 评估，返回排名表。

    Parameters
    ----------
    features_df : 特征 DataFrame (index=Date)
    forward_ret : 未来 N 日收益率 (index=Date)
    ic_window : 滚动 IC 的窗口
    min_non_null : 因子最低非空比例，低于此则跳过

    Returns
    -------
    pd.DataFrame: 按 |ICIR| 降序排列的因子评估表
    """
    # 对齐 index
    common_idx = features_df.index.intersection(forward_ret.index)
    feat = features_df.loc[common_idx]
    ret = forward_ret.loc[common_idx]

    results = []
    for col in feat.columns:
        series = feat[col]
        non_null_ratio = series.notna().mean()
        if non_null_ratio < min_non_null:
            continue
        summary = factor_ic_summary(series, ret, window=ic_window)
        summary["factor"] = col
        summary["non_null_ratio"] = round(non_null_ratio, 3)
        results.append(summary)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["abs_ICIR"] = df["ICIR"].abs()
    df = df.sort_values("abs_ICIR", ascending=False).reset_index(drop=True)
    df = df[["factor", "IC_mean", "IC_std", "ICIR", "abs_ICIR",
             "IC_pos_ratio", "IC_abs_mean", "non_null_ratio", "N"]]
    return df


def ic_decay_analysis(factor: pd.Series, price: pd.Series,
                      horizons: list[int] = None) -> pd.DataFrame:
    """
    因子衰减分析: 分别看因子对 1d/2d/5d/10d/20d 收益的 IC。
    用于判断因子在哪个预测周期最有效。

    Parameters
    ----------
    factor : 因子值
    price : 收盘价 (用于计算不同 horizon 的 forward return)
    horizons : 预测周期列表

    Returns
    -------
    pd.DataFrame: 每个 horizon 的 IC 统计
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 20]

    results = []
    for h in horizons:
        fwd_ret = price.pct_change(h).shift(-h)
        summary = factor_ic_summary(factor, fwd_ret, window=60)
        summary["horizon"] = f"{h}d"
        results.append(summary)

    return pd.DataFrame(results).set_index("horizon")
