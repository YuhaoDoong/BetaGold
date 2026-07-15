"""
技术面特征工程 (快通道, 日频)

基于 GLD 价格/成交量 + 关联品种构建技术指标。
输入: data/raw/market/*.csv
输出: 日频特征 DataFrame
"""

import os
import logging

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class TechnicalFeatureBuilder:
    """技术面特征构建器"""

    def __init__(self, config: dict = None, ticker: str = "gld"):
        self.config = config or load_config()
        self.raw_path = self.config["paths"]["raw_data"]
        self.ticker = ticker.lower()

    def _load_market_data(self) -> dict[str, pd.DataFrame]:
        """加载所有市场数据"""
        market_dir = os.path.join(self.raw_path, "market")
        data = {}
        for fname in os.listdir(market_dir):
            if fname.endswith(".csv"):
                name = fname.replace(".csv", "")
                df = pd.read_csv(
                    os.path.join(market_dir, fname),
                    index_col=0, parse_dates=True,
                )
                data[name] = df
        logger.info(f"加载 {len(data)} 个市场数据文件")
        return data

    def build(self) -> pd.DataFrame:
        """构建全部技术面特征"""
        market = self._load_market_data()

        primary = market.get(self.ticker)
        if primary is None:
            raise FileNotFoundError(f"{self.ticker.upper()} 数据未找到")

        features = pd.DataFrame(index=primary.index)
        features.index.name = "Date"

        # 1. 收益率
        features = self._add_returns(features, primary)

        # 2. 移动均线 & 趋势
        features = self._add_moving_averages(features, primary)

        # 3. 动量指标 (RSI / MACD)
        features = self._add_momentum(features, primary)

        # 4. 波动率指标 (Bollinger / ATR / 历史波动率)
        features = self._add_volatility_indicators(features, primary)

        # 5. 成交量特征
        features = self._add_volume_features(features, primary)

        # 6. 跳空/缺口
        features = self._add_gap_features(features, primary)

        # 7. 关联品种特征
        features = self._add_cross_asset_features(features, market)

        # 8. Alpha158 因子 (借鉴 Qlib)
        features = self._add_alpha158_factors(features, primary)

        logger.info(f"技术面特征: {features.shape[1]} 列, "
                    f"{features.shape[0]} 行")
        return features

    def _add_returns(self, features: pd.DataFrame,
                     gld: pd.DataFrame) -> pd.DataFrame:
        """收益率特征"""
        close = gld["Close"]

        # 日收益率
        features["ret_1d"] = close.pct_change()

        # 多窗口累计收益
        for w in [2, 3, 5, 10, 20, 60]:
            features[f"ret_{w}d"] = close.pct_change(w)

        return features

    def _add_moving_averages(self, features: pd.DataFrame,
                             gld: pd.DataFrame) -> pd.DataFrame:
        """移动均线 & 趋势特征"""
        close = gld["Close"]

        # SMA (只输出相对偏离度，不输出原始价位)
        sma = {}
        for w in [5, 10, 20, 60, 120]:
            sma[w] = close.rolling(w).mean()
            features[f"close_to_sma_{w}"] = (close - sma[w]) / sma[w]

        # 均线斜率 (5日变化率)
        features["sma_20_slope"] = sma[20].pct_change(5)
        features["sma_60_slope"] = sma[60].pct_change(5)

        # 均线排列: 短期均线在长期之上的数量
        features["ma_alignment"] = (
            (sma[5] > sma[10]).astype(int) +
            (sma[10] > sma[20]).astype(int) +
            (sma[20] > sma[60]).astype(int)
        )

        return features

    def _add_momentum(self, features: pd.DataFrame,
                      gld: pd.DataFrame) -> pd.DataFrame:
        """动量指标"""
        close = gld["Close"]

        # RSI (14日)
        for period in [7, 14]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_hist"] = macd_line - signal_line

        # 随机指标 %K
        for period in [14]:
            low_min = gld["Low"].rolling(period).min()
            high_max = gld["High"].rolling(period).max()
            denom = (high_max - low_min).replace(0, np.nan)
            features[f"stoch_k_{period}"] = (
                (close - low_min) / denom * 100
            )
            features[f"stoch_d_{period}"] = (
                features[f"stoch_k_{period}"].rolling(3).mean()
            )

        return features

    def _add_volatility_indicators(self, features: pd.DataFrame,
                                   gld: pd.DataFrame) -> pd.DataFrame:
        """波动率指标"""
        close = gld["Close"]
        high = gld["High"]
        low = gld["Low"]

        # 历史波动率 (年化) — 只保留 5d/60d, 10d/20d 由 vol_features 提供
        log_ret = np.log(close / close.shift(1))
        for w in [5, 60]:
            features[f"hv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

        # Bollinger Bands (20, 2) — 只输出相对指标
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        features["bb_width"] = (bb_upper - bb_lower) / sma20
        features["bb_position"] = (
            (close - bb_lower) /
            (bb_upper - bb_lower).replace(0, np.nan)
        )

        # ATR (14日)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean()
        features["atr_14_pct"] = features["atr_14"] / close

        # 波动率变化率 (hv_5d 的 5 日变化)
        features["hv_5d_change"] = features["hv_5d"].pct_change(5)

        # 高低幅
        features["daily_range_pct"] = (high - low) / close

        return features

    def _add_volume_features(self, features: pd.DataFrame,
                             gld: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        volume = gld["Volume"]

        # 成交量均线比
        for w in [5, 20]:
            vol_ma = volume.rolling(w).mean()
            features[f"vol_ratio_{w}d"] = volume / vol_ma.replace(0, np.nan)

        # 成交量变化率
        features["vol_change_1d"] = volume.pct_change()

        # OBV 方向
        obv_direction = np.sign(gld["Close"].diff())
        features["obv_direction"] = obv_direction

        # 量价配合: 价涨量增=1, 价涨量缩=-1, etc
        price_up = (gld["Close"].diff() > 0).astype(int)
        vol_up = (volume.diff() > 0).astype(int)
        features["price_vol_confirm"] = (
            (price_up * 2 - 1) * (vol_up * 2 - 1)
        )

        return features

    def _add_gap_features(self, features: pd.DataFrame,
                          gld: pd.DataFrame) -> pd.DataFrame:
        """跳空/缺口特征"""
        # 跳空幅度
        gap = gld["Open"] - gld["Close"].shift(1)
        gap_pct = gap / gld["Close"].shift(1)
        features["gap_pct"] = gap_pct

        # 跳空方向
        features["gap_up"] = (gap_pct > 0.005).astype(int)
        features["gap_down"] = (gap_pct < -0.005).astype(int)

        return features

    def _add_cross_asset_features(self, features: pd.DataFrame,
                                  market: dict) -> pd.DataFrame:
        """关联品种特征 (以 self.ticker 的 Close 作为基准对齐)"""
        base_close = market[self.ticker]["Close"]

        # 期货/ETF 基差 (黄金: GC/GLD; 白银: SI/SLV)
        # 保留 gc_gld_ratio 名称以兼容下游代码 (包括 SLV 模型, 命名含义为
        # "本资产的期货/ETF 基差")
        fut_key = "gold_futures" if self.ticker == "gld" else "silver"
        if fut_key in market:
            fc = market[fut_key]["Close"].reindex(base_close.index, method="ffill")
            ratio = fc / base_close
            features["gc_gld_ratio"] = ratio
            features["gc_gld_ratio_zscore"] = (
                (ratio - ratio.rolling(60).mean()) /
                ratio.rolling(60).std().replace(0, np.nan)
            )

        # 美元指数
        if "dxy" in market:
            dxy = market["dxy"]["Close"].reindex(base_close.index, method="ffill")
            features["dxy_ret_1d"] = dxy.pct_change()
            features["dxy_ret_5d"] = dxy.pct_change(5)
            features["dxy_sma20_dev"] = (
                (dxy - dxy.rolling(20).mean()) /
                dxy.rolling(20).mean()
            )

        # VIX
        if "vix" in market:
            vix = market["vix"]["Close"].reindex(base_close.index, method="ffill")
            features["vix_level"] = vix
            features["vix_ret_1d"] = vix.pct_change()
            features["vix_sma20_dev"] = (
                (vix - vix.rolling(20).mean()) /
                vix.rolling(20).mean()
            )

        # 铜金比 (经济风向标)
        if "copper" in market and "gold_futures" in market:
            hg = market["copper"]["Close"].reindex(base_close.index, method="ffill")
            gc = market["gold_futures"]["Close"].reindex(base_close.index, method="ffill")
            cu_au = hg / gc.replace(0, np.nan)
            features["copper_gold_ratio"] = cu_au
            features["copper_gold_ratio_change"] = cu_au.pct_change(20)

        # 白银/黄金比
        if "silver" in market and "gold_futures" in market:
            si = market["silver"]["Close"].reindex(base_close.index, method="ffill")
            gc = market["gold_futures"]["Close"].reindex(base_close.index, method="ffill")
            features["gold_silver_ratio"] = gc / si.replace(0, np.nan)

        # 10Y 国债收益率
        if "us10y" in market:
            tnx = market["us10y"]["Close"].reindex(base_close.index, method="ffill")
            features["us10y_level"] = tnx
            features["us10y_change_5d"] = tnx.diff(5)

        # 原油
        if "crude_oil" in market:
            cl = market["crude_oil"]["Close"].reindex(base_close.index, method="ffill")
            features["crude_ret_5d"] = cl.pct_change(5)

        return features

    # ------------------------------------------------------------------
    # Alpha158 因子 (借鉴 Microsoft Qlib)
    # ------------------------------------------------------------------
    def _add_alpha158_factors(self, features: pd.DataFrame,
                              gld: pd.DataFrame) -> pd.DataFrame:
        """
        从 Qlib Alpha158 中提取我们原本缺失的因子。
        全部用 pandas 实现，不依赖 Qlib 框架。
        """
        close = gld["Close"]
        open_ = gld["Open"]
        high = gld["High"]
        low = gld["Low"]
        volume = gld["Volume"]

        # ---- K线形态因子 ----
        hl_range = (high - low).replace(0, np.nan)
        features["kbar_kmid"] = (close - open_) / open_              # 实体大小
        features["kbar_klen"] = (high - low) / open_                 # K线长度
        features["kbar_kmid2"] = (close - open_) / hl_range          # 实体/全长
        features["kbar_kup"] = (high - np.maximum(open_, close)) / open_   # 上影线
        features["kbar_kup2"] = (high - np.maximum(open_, close)) / hl_range
        features["kbar_klow"] = (np.minimum(open_, close) - low) / open_   # 下影线
        features["kbar_klow2"] = (np.minimum(open_, close) - low) / hl_range
        features["kbar_ksft"] = (2 * close - high - low) / open_    # 重心偏移
        features["kbar_ksft2"] = (2 * close - high - low) / hl_range

        windows = [5, 10, 20, 30, 60]

        for w in windows:
            # ---- 趋势线性因子 ----
            # BETA: 线性回归斜率 / close (趋势强度)
            features[f"beta_{w}d"] = (
                close.rolling(w).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
                ) / close
            )

            # RSQR: 线性回归 R² (趋势线性度)
            def _rsqr(x):
                n = len(x)
                if n < 3:
                    return np.nan
                t = np.arange(n, dtype=float)
                p = np.polyfit(t, x, 1)
                fitted = np.polyval(p, t)
                ss_res = np.sum((x - fitted) ** 2)
                ss_tot = np.sum((x - x.mean()) ** 2)
                if ss_tot == 0:
                    return np.nan
                return 1.0 - ss_res / ss_tot

            features[f"rsqr_{w}d"] = close.rolling(w).apply(_rsqr, raw=True)

            # RESI: 线性回归残差 / close (偏离趋势线)
            def _resi(x):
                n = len(x)
                if n < 3:
                    return np.nan
                t = np.arange(n, dtype=float)
                p = np.polyfit(t, x, 1)
                fitted = np.polyval(p, t)
                return (x[-1] - fitted[-1])

            features[f"resi_{w}d"] = (
                close.rolling(w).apply(_resi, raw=True) / close
            )

            # ---- 价格位置因子 ----
            # RSV: 价格在 N 日区间中的位置 (类似随机指标)
            low_min = low.rolling(w).min()
            high_max = high.rolling(w).max()
            rsv_denom = (high_max - low_min).replace(0, np.nan)
            features[f"rsv_{w}d"] = (close - low_min) / rsv_denom

            # QTLU / QTLD: 80%/20% 分位数位置
            features[f"qtlu_{w}d"] = (
                close.rolling(w).quantile(0.8) / close
            )
            features[f"qtld_{w}d"] = (
                close.rolling(w).quantile(0.2) / close
            )

            # ---- 高低点时序因子 (Aroon 类) ----
            # IMAX: 最近 N 日内最高价出现在第几天 (归一化)
            features[f"imax_{w}d"] = (
                high.rolling(w).apply(lambda x: np.argmax(x), raw=True) / w
            )
            # IMIN: 最近 N 日内最低价出现在第几天
            features[f"imin_{w}d"] = (
                low.rolling(w).apply(lambda x: np.argmin(x), raw=True) / w
            )
            # IMXD: 高点比低点晚多少天 (正=下行动量)
            features[f"imxd_{w}d"] = (
                features[f"imax_{w}d"] - features[f"imin_{w}d"]
            )

            # ---- 量价相关因子 ----
            # CORR: 价格与 log(成交量) 的相关系数
            log_vol = np.log1p(volume)
            features[f"corr_{w}d"] = close.rolling(w).corr(log_vol)

            # CORD: 收益率与成交量变化率的相关系数
            ret_1d = close.pct_change()
            vol_ret = volume.pct_change()
            features[f"cord_{w}d"] = ret_1d.rolling(w).corr(vol_ret)

            # ---- 涨跌统计因子 ----
            up = (close > close.shift(1)).astype(float)
            down = (close < close.shift(1)).astype(float)

            # CNTP / CNTN: N 日内上涨/下跌天数比例
            features[f"cntp_{w}d"] = up.rolling(w).mean()
            features[f"cntn_{w}d"] = down.rolling(w).mean()
            # CNTD: 上涨比例 - 下跌比例
            features[f"cntd_{w}d"] = (
                features[f"cntp_{w}d"] - features[f"cntn_{w}d"]
            )

            # SUMP / SUMN: 涨幅总量 / 绝对总变动 (类 RSI)
            diff = close.diff()
            abs_diff_sum = diff.abs().rolling(w).sum().replace(0, np.nan)
            features[f"sump_{w}d"] = (
                diff.clip(lower=0).rolling(w).sum() / abs_diff_sum
            )
            features[f"sumn_{w}d"] = (
                (-diff).clip(lower=0).rolling(w).sum() / abs_diff_sum
            )
            # SUMD: 净涨跌方向强度
            features[f"sumd_{w}d"] = (
                features[f"sump_{w}d"] - features[f"sumn_{w}d"]
            )

            # ---- 成交量统计因子 ----
            # VMA: N日成交量均值 / 当日成交量
            vol_safe = volume.replace(0, np.nan)
            features[f"vma_{w}d"] = volume.rolling(w).mean() / vol_safe
            # VSTD: N日成交量标准差 / 当日成交量
            features[f"vstd_{w}d"] = volume.rolling(w).std() / vol_safe

        logger.info(f"Alpha158 因子: 添加完成")
        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    builder = TechnicalFeatureBuilder()
    df = builder.build()
    print(f"\n技术面特征: {df.shape[1]} 列, {df.shape[0]} 行")
    print(f"列名: {list(df.columns)}")
    print(f"\n缺失率:")
    missing = df.isnull().mean()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
