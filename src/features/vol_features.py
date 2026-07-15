"""
波动率特征工程 (日频)

基于 GVZ、VIX 期限结构、MOVE、经济事件日历构建波动率特征。
期权交易的核心 -- 方向看对但 IV 塌了一样亏钱。

输入: data/raw/volatility/, data/raw/events/, data/raw/market/
输出: 日频特征 DataFrame
"""

import os
import logging

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class VolFeatureBuilder:
    """波动率特征构建器"""

    def __init__(self, config: dict = None, ticker: str = "gld"):
        self.config = config or load_config()
        self.raw_path = self.config["paths"]["raw_data"]
        self.ticker = ticker.lower()

    def _load_price(self) -> pd.DataFrame:
        path = os.path.join(self.raw_path, "market", f"{self.ticker}.csv")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    # 兼容旧调用
    def _load_gld(self) -> pd.DataFrame:
        return self._load_price()

    def _load_gvz(self) -> pd.Series:
        path = os.path.join(self.raw_path, "volatility", "gvz.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["GVZ"]

    def _load_vix_term(self) -> pd.DataFrame:
        path = os.path.join(self.raw_path, "volatility",
                            "vix_term_structure.csv")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _load_move(self) -> pd.Series:
        path = os.path.join(self.raw_path, "volatility", "move.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    def _load_events(self) -> pd.DataFrame:
        path = os.path.join(self.raw_path, "events",
                            "economic_calendar.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["Date"])

    def build(self) -> pd.DataFrame:
        """构建全部波动率特征"""
        price = self._load_price()
        features = pd.DataFrame(index=price.index)
        features.index.name = "Date"

        # 1. GVZ 特征 (黄金 IV 指数, 对白银仍有参考价值)
        features = self._add_gvz_features(features)

        # 2. VIX 期限结构特征
        features = self._add_vix_term_features(features)

        # 3. MOVE (债券波动率)
        features = self._add_move_features(features)

        # 4. 已实现波动率 vs 隐含波动率 (RV 用本资产价格)
        features = self._add_rv_iv_features(features, price)

        # 5. 事件窗口特征
        features = self._add_event_features(features)

        logger.info(f"波动率特征: {features.shape[1]} 列, "
                    f"{features.shape[0]} 行")
        return features

    def _add_gvz_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """GVZ 黄金波动率指数特征"""
        gvz = self._load_gvz()
        gvz = gvz.reindex(features.index, method="ffill")

        features["gvz"] = gvz
        features["gvz_ret_1d"] = gvz.pct_change()
        features["gvz_ret_5d"] = gvz.pct_change(5)

        # GVZ 历史分位数 (1年窗口)
        features["gvz_pctile_252d"] = gvz.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )

        # GVZ SMA 偏离
        gvz_sma20 = gvz.rolling(20).mean()
        features["gvz_sma20_dev"] = (gvz - gvz_sma20) / gvz_sma20

        # GVZ 是否处于高/低区间
        features["gvz_high"] = (
            features["gvz_pctile_252d"] > 0.8
        ).astype(int)
        features["gvz_low"] = (
            features["gvz_pctile_252d"] < 0.2
        ).astype(int)

        return features

    def _add_vix_term_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """VIX 期限结构特征"""
        vix_term = self._load_vix_term()
        vix_term = vix_term.reindex(features.index, method="ffill")

        # VIX 各期限 (VIX 本身已在 tech features 中作为 vix_level)
        for col in ["VIX9D", "VIX3M", "VIX6M"]:
            if col in vix_term.columns:
                features[f"vix_{col.lower()}"] = vix_term[col]

        # 期限结构斜率 (contango vs backwardation)
        if "VIX" in vix_term.columns and "VIX3M" in vix_term.columns:
            features["vix_term_slope"] = (
                vix_term["VIX3M"] - vix_term["VIX"]
            )
            # 正值 = contango (正常), 负值 = backwardation (恐慌)
            features["vix_backwardation"] = (
                features["vix_term_slope"] < 0
            ).astype(int)

        if "VIX" in vix_term.columns and "VIX6M" in vix_term.columns:
            features["vix_term_slope_6m"] = (
                vix_term["VIX6M"] - vix_term["VIX"]
            )

        # 短期 vs 超短期 (9D vs VIX)
        if "VIX9D" in vix_term.columns and "VIX" in vix_term.columns:
            features["vix_9d_vs_30d"] = (
                vix_term["VIX9D"] - vix_term["VIX"]
            )

        return features

    def _add_move_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """MOVE 债券波动率特征"""
        move = self._load_move()
        move = move.reindex(features.index, method="ffill")

        features["move"] = move
        features["move_ret_5d"] = move.pct_change(5)

        # MOVE 历史分位数
        features["move_pctile_252d"] = move.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )

        return features

    def _add_rv_iv_features(self, features: pd.DataFrame,
                            gld: pd.DataFrame) -> pd.DataFrame:
        """已实现波动率 vs 隐含波动率"""
        close = gld["Close"]
        log_ret = np.log(close / close.shift(1))

        # 已实现波动率 (年化, ×100 对齐 GVZ 百分比格式)
        rv = {}
        for w in [5, 10, 20]:
            rv[w] = log_ret.rolling(w).std() * np.sqrt(252) * 100
            features[f"rv_{w}d"] = rv[w]

        # GVZ (隐含波动率) vs 已实现波动率 = 波动率风险溢价
        gvz = self._load_gvz().reindex(features.index, method="ffill")
        if not gvz.empty:
            features["vrp_20d"] = gvz - rv[20]
            features["vrp_10d"] = gvz - rv[10]

            # 波动率风险溢价分位数
            features["vrp_20d_pctile"] = features["vrp_20d"].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False,
            )

        return features

    def _add_event_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """经济事件窗口特征"""
        events = self._load_events()
        if events.empty:
            return features

        for event_type in ["FOMC", "NFP", "CPI"]:
            event_dates = events[events["Event"] == event_type]["Date"].values

            if len(event_dates) == 0:
                continue

            # 距下一个事件天数
            col = f"days_to_{event_type.lower()}"
            days_list = []
            for date in features.index:
                future = event_dates[event_dates > date]
                if len(future) > 0:
                    days_list.append(
                        (pd.Timestamp(future[0]) - date).days
                    )
                else:
                    days_list.append(np.nan)
            features[col] = days_list

            # 事件前3天窗口
            features[f"{event_type.lower()}_window_3d"] = (
                features[col].between(0, 3)
            ).astype(int)

            # 是否当天
            features[f"is_{event_type.lower()}_day"] = (
                features.index.isin(event_dates)
            ).astype(int)

        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    builder = VolFeatureBuilder()
    df = builder.build()
    print(f"\n波动率特征: {df.shape[1]} 列, {df.shape[0]} 行")
    print(f"列名: {list(df.columns)}")
    print(f"\n缺失率:")
    missing = df.isnull().mean()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
