"""
宏观 Regime 分类器 (Phase 4A Step 2)

基于多个宏观因子综合判断当前市场处于:
- Bull (看多): 宽松周期 + 实际利率偏低 + USD 走弱 + 央行买入
- Bear (看空): 紧缩周期 + 实际利率偏高 + USD 走强
- Mixed (震荡): 因子方向不一致 / 转折期

不使用 ML 分类器, 而是用规则 + 打分系统, 原因:
1. Regime 标签没有客观标准 (无法训练有监督模型)
2. 规则透明可解释, 便于调整
3. 宏观因子变化缓慢, 不需要复杂非线性模型
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    多因子打分 Regime 分类器。

    对每个因子计算牛/熊分数 (-1 到 +1), 加权汇总后分类。
    """

    # 因子权重 (基于 Step 2 实测调参)
    # 加入价格动量: 2011-2015 金价大跌期间纯宏观因子未能识别 Bear
    FACTOR_WEIGHTS = {
        "price_momentum": 0.25,       # 价格趋势 (最直接的信号)
        "fed_rate_direction": 0.20,   # 利率方向
        "usd_trend": 0.15,           # 美元趋势
        "central_bank": 0.15,        # 央行购金 (从0.20降, 2013年误导)
        "risk_sentiment": 0.10,      # 风险情绪 (GVZ)
        "inflation": 0.10,           # 通胀趋势
        "real_yield_level": 0.05,    # 实际利率 (弱因子)
    }

    def __init__(self, bull_threshold: float = 0.2,
                 bear_threshold: float = -0.2,
                 smooth_window: int = 60,
                 min_hold_days: int = 20):
        """
        bull_threshold: 综合分数 > 此值 → Bull
        bear_threshold: 综合分数 < 此值 → Bear
        smooth_window: EMA 平滑窗口 (天), 避免频繁跳动
        min_hold_days: 最小持有天数, regime 确认后至少持续 N 天
        """
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.smooth_window = smooth_window
        self.min_hold_days = min_hold_days
        self.name = "RegimeClassifier"

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        对每一行特征计算 regime 分类。

        返回 DataFrame: regime_score, regime (Bull/Bear/Mixed),
                       以及各因子子分数。
        """
        n = len(features)
        scores = pd.DataFrame(index=features.index)

        # 0. 价格动量: GLD 60d 收益率
        #    上涨趋势 → 利好 (+1), 下跌趋势 → 利空 (-1)
        if "ret_60d" in features.columns:
            ret60 = features["ret_60d"]
            # 60d 收益超过 ±5% 为强趋势, ±10% 饱和
            scores["price_momentum"] = np.clip(ret60 / 0.10, -1, 1)
        elif "ret_20d" in features.columns:
            ret20 = features["ret_20d"]
            scores["price_momentum"] = np.clip(ret20 / 0.05, -1, 1)
        else:
            scores["price_momentum"] = 0.0

        # 1. 利率方向: fed_funds_rate 60d 变化
        #    降息 → 利好黄金 (+1), 加息 → 利空 (-1)
        if "fed_funds_rate_change_60d" in features.columns:
            dff_chg = features["fed_funds_rate_change_60d"]
            scores["fed_rate_direction"] = np.clip(-dff_chg / 0.5, -1, 1)
        else:
            scores["fed_rate_direction"] = 0.0

        # 2. 实际利率水平: real_yield_10y
        #    低利率 → 利好黄金 (+1), 高利率 → 利空 (-1)
        #    用 zscore 做归一化
        if "real_yield_10y_zscore" in features.columns:
            ry_z = features["real_yield_10y_zscore"]
            scores["real_yield_level"] = np.clip(-ry_z / 2, -1, 1)
        elif "real_yield_10y" in features.columns:
            ry = features["real_yield_10y"]
            # 历史中位数约 0.5%, 用 1% 作为标准差近似
            scores["real_yield_level"] = np.clip(-(ry - 0.5) / 1.0, -1, 1)
        else:
            scores["real_yield_level"] = 0.0

        # 3. 美元趋势: tw_usd 20d 收益率
        #    美元走弱 → 利好黄金 (+1), 走强 → 利空 (-1)
        if "tw_usd_ret_20d" in features.columns:
            usd_ret = features["tw_usd_ret_20d"]
            scores["usd_trend"] = np.clip(-usd_ret / 0.02, -1, 1)
        elif "tw_usd_zscore" in features.columns:
            scores["usd_trend"] = np.clip(
                -features["tw_usd_zscore"] / 2, -1, 1)
        else:
            scores["usd_trend"] = 0.0

        # 4. 央行购金: 12m rolling
        #    持续大量购金 → 利好 (+1), 减少 → 利空 (-1)
        if "cb_global_12m_rolling" in features.columns:
            cb = features["cb_global_12m_rolling"]
            # 历史范围大约 -200 ~ 800, 中位数约 200
            scores["central_bank"] = np.clip((cb - 200) / 300, -1, 1)
        else:
            scores["central_bank"] = 0.0

        # 5. 风险情绪: GVZ 分位数
        #    GVZ 偏高 → 避险 → 轻微利好黄金, 但极端高也意味不确定
        #    用 U 形: 中等水平 → 0, 偏高 → 轻微正, 极高 → 回到 0
        if "gvz_pctile_252d" in features.columns:
            gvz_p = features["gvz_pctile_252d"].fillna(0.5)
            # 高 GVZ (>75 pctile) → 避险情绪, 轻微利好黄金
            # 低 GVZ (<25 pctile) → 平静, 中性
            scores["risk_sentiment"] = np.clip((gvz_p - 0.5) / 0.3, -1, 1)
        else:
            scores["risk_sentiment"] = 0.0

        # 6. 通胀趋势
        #    高通胀 → 利好黄金 (保值需求), 低通胀 → 利空
        if "cpi_yoy" in features.columns:
            cpi = features["cpi_yoy"]
            # 2% 为中性, 高于 3% 利好, 低于 1% 利空
            scores["inflation"] = np.clip((cpi - 0.02) / 0.02, -1, 1)
        elif "breakeven_10y" in features.columns:
            be = features["breakeven_10y"]
            scores["inflation"] = np.clip((be - 2.0) / 0.5, -1, 1)
        else:
            scores["inflation"] = 0.0

        # 加权汇总
        composite = pd.Series(0.0, index=features.index)
        for factor, weight in self.FACTOR_WEIGHTS.items():
            if factor in scores.columns:
                composite += scores[factor].fillna(0) * weight

        scores["regime_score_raw"] = composite

        # EMA 平滑 (避免日频噪声)
        smoothed = composite.ewm(span=self.smooth_window, min_periods=20).mean()
        scores["regime_score"] = smoothed

        # 基于平滑分数分类
        raw_regime = pd.Series("Mixed", index=features.index)
        raw_regime[smoothed > self.bull_threshold] = "Bull"
        raw_regime[smoothed < self.bear_threshold] = "Bear"

        # 最小持有期: 一旦确认 regime, 至少持续 min_hold_days
        if self.min_hold_days > 1:
            regime = self._apply_min_hold(raw_regime)
        else:
            regime = raw_regime

        scores["regime"] = regime
        return scores

    def _apply_min_hold(self, raw_regime: pd.Series,
                        min_days: int = None) -> pd.Series:
        """
        最小持有期过滤: 如果新 regime 持续不到 min_days 天,
        回退到之前的 regime。
        """
        min_days = min_days or self.min_hold_days
        result = raw_regime.copy()
        values = result.values.copy()
        n = len(values)

        current = values[0]

        i = 1
        while i < n:
            if values[i] != current:
                # 找到新 regime 的持续长度
                new_regime = values[i]
                j = i
                while j < n and values[j] == new_regime:
                    j += 1
                duration = j - i

                if duration < min_days:
                    # 持续不够, 回退
                    values[i:j] = current
                    i = j
                else:
                    # 确认切换
                    current = new_regime
                    hold_start = i
                    i = j
            else:
                i += 1

        return pd.Series(values, index=raw_regime.index)

    def classify_simple(self, features: pd.DataFrame) -> pd.Series:
        """只返回 regime 列"""
        return self.classify(features)["regime"]
