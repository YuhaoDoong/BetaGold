"""
预测标签构建

基于 GLD 价格构建各类预测标签 (label / target)。
标签定义与 README 第七章一致。

输入: data/raw/market/gld.csv
输出: 日频标签 DataFrame
"""

import os
import logging

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class LabelBuilder:
    """预测标签构建器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_path = self.config["paths"]["raw_data"]

    def _load_gld(self) -> pd.DataFrame:
        path = os.path.join(self.raw_path, "market", "gld.csv")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def build(self) -> pd.DataFrame:
        """构建全部标签"""
        gld = self._load_gld()
        close = gld["Close"]

        labels = pd.DataFrame(index=gld.index)
        labels.index.name = "Date"

        # === 1. 收益率标签 (fwd_ 前缀避免与特征列名冲突) ===
        for horizon in [5, 10, 20]:
            labels[f"fwd_ret_{horizon}d"] = close.pct_change(horizon).shift(-horizon)

        # === 2. 方向标签 ===
        # 涨 (>+1%), 跌 (<-1%), 震荡
        for horizon in [5, 10]:
            ret = labels[f"fwd_ret_{horizon}d"]
            labels[f"direction_{horizon}d"] = pd.cut(
                ret,
                bins=[-np.inf, -0.01, 0.01, np.inf],
                labels=["down", "flat", "up"],
            )

        # === 3. 幅度标签 ===
        ret_10 = labels["fwd_ret_10d"].abs()
        labels["magnitude_10d"] = pd.cut(
            ret_10,
            bins=[-np.inf, 0.02, 0.05, np.inf],
            labels=["small", "medium", "large"],
        )

        # === 4. 已实现波动率标签 ===
        log_ret = np.log(close / close.shift(1))
        for horizon in [10, 20]:
            # 未来 N 日实现波动率 (年化)
            rv = log_ret.rolling(horizon).std() * np.sqrt(252)
            labels[f"fwd_rv_{horizon}d"] = rv.shift(-horizon)

        # === 5. 波动状态标签 ===
        rv_20 = log_ret.rolling(20).std() * np.sqrt(252)
        rv_20_pctile = rv_20.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )
        labels["vol_regime"] = pd.cut(
            rv_20_pctile,
            bins=[-np.inf, 0.33, 0.67, np.inf],
            labels=["low_vol", "normal", "high_vol"],
        )

        # === 6. IV-RV 价差 ===
        # 需要 GVZ 数据
        gvz_path = os.path.join(self.raw_path, "volatility", "gvz.csv")
        if os.path.exists(gvz_path):
            gvz = pd.read_csv(gvz_path, index_col=0, parse_dates=True)
            gvz = gvz["GVZ"].reindex(close.index, method="ffill")
            # GVZ 是百分比格式 (如 35.31 = 35.31%)
            # fwd_rv 是年化小数 (如 0.25 = 25%)
            rv_20_future = labels["fwd_rv_20d"]
            labels["iv_rv_spread"] = gvz / 100 - rv_20_future

        # === 7. 尾部风险标签 ===
        # 未来 10 天内是否出现 > 3sigma 的日移动
        daily_std = log_ret.rolling(60).std()
        abs_log_ret = log_ret.abs()
        tail_threshold = daily_std * 3

        # 向前看 10 天内是否有尾部事件
        tail_event = pd.Series(False, index=close.index)
        for shift_d in range(1, 11):
            future_ret = abs_log_ret.shift(-shift_d)
            future_threshold = tail_threshold  # 用当前的阈值
            tail_event = tail_event | (future_ret > future_threshold)
        labels["tail_event_flag"] = tail_event.astype(int)

        # === 8. 最大回撤 (未来 10 日) ===
        max_dd_list = []
        for i in range(len(close)):
            if i + 10 >= len(close):
                max_dd_list.append(np.nan)
                continue
            future_prices = close.iloc[i:i + 11]
            cum_max = future_prices.cummax()
            drawdown = (future_prices - cum_max) / cum_max
            max_dd_list.append(drawdown.min())
        labels["max_dd_10d"] = max_dd_list

        # 统计
        n_labels = labels.shape[1]
        n_valid = labels.dropna(how="all").shape[0]
        logger.info(f"标签: {n_labels} 个, {n_valid} 行有效")
        return labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    builder = LabelBuilder()
    df = builder.build()
    print(f"\n标签: {df.shape[1]} 列, {df.shape[0]} 行")
    print(f"列名: {list(df.columns)}")

    # 数值型标签统计
    numeric = df.select_dtypes(include=[np.number])
    print(f"\n数值型标签统计:")
    print(numeric.describe().round(4))

    # 分类标签分布
    for col in df.select_dtypes(include=["category"]).columns:
        print(f"\n{col}:")
        print(df[col].value_counts())
