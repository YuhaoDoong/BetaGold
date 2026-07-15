"""
宏观面特征工程 (慢通道, 月/季频 -> 日频对齐)

基于 FRED 宏观数据 + 央行购金 + COT 持仓构建宏观因子。
关键: 必须按 release date 对齐，防止未来函数泄漏。

输入: data/raw/macro/, data/raw/central_bank/, data/raw/cot/
输出: 日频特征 DataFrame (对齐到 GLD 交易日)
"""

import os
import logging

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class MacroFeatureBuilder:
    """宏观面特征构建器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_path = self.config["paths"]["raw_data"]

    def _load_macro_panel(self) -> pd.DataFrame:
        """加载宏观面板"""
        path = os.path.join(self.raw_path, "macro", "macro_panel.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"宏观面板: {df.shape}")
        return df

    def _load_cot(self) -> pd.DataFrame:
        """加载 COT 持仓数据"""
        path = os.path.join(self.raw_path, "cot", "gold_cot.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        logger.info(f"COT 数据: {df.shape}")
        return df

    def _load_cb_features(self) -> pd.DataFrame:
        """加载央行购金特征"""
        path = os.path.join(self.raw_path, "central_bank", "cb_features.csv")
        if not os.path.exists(path):
            logger.warning("央行购金特征文件不存在")
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"央行购金特征: {df.shape}")
        return df

    def _load_gld_dates(self) -> pd.DatetimeIndex:
        """加载 GLD 交易日作为对齐基准"""
        path = os.path.join(self.raw_path, "market", "gld.csv")
        gld = pd.read_csv(path, index_col=0, parse_dates=True)
        return gld.index

    def build(self) -> pd.DataFrame:
        """构建全部宏观面特征"""
        gld_dates = self._load_gld_dates()
        features = pd.DataFrame(index=gld_dates)
        features.index.name = "Date"

        # 1. FRED 宏观因子
        features = self._add_macro_factors(features)

        # 2. COT 持仓特征
        features = self._add_cot_features(features)

        # 3. 央行购金特征
        features = self._add_cb_features(features)

        logger.info(f"宏观面特征: {features.shape[1]} 列, "
                    f"{features.shape[0]} 行")
        return features

    def _add_macro_factors(self, features: pd.DataFrame) -> pd.DataFrame:
        """FRED 宏观因子特征"""
        panel = self._load_macro_panel()

        # 对齐到 GLD 交易日 (ffill)
        panel = panel.reindex(features.index, method="ffill")

        # --- 实际利率相关 ---
        if "DFII10" in panel.columns:
            features["real_yield_10y"] = panel["DFII10"]
            features["real_yield_10y_change_20d"] = panel["DFII10"].diff(20)
            features["real_yield_10y_zscore"] = (
                (panel["DFII10"] - panel["DFII10"].rolling(252).mean()) /
                panel["DFII10"].rolling(252).std().replace(0, np.nan)
            )

        if "DFII5" in panel.columns:
            features["real_yield_5y"] = panel["DFII5"]

        # 实际利率曲线 (10Y - 5Y)
        if "DFII10" in panel.columns and "DFII5" in panel.columns:
            features["real_yield_curve"] = panel["DFII10"] - panel["DFII5"]

        # --- 通胀预期 ---
        if "T10YIE" in panel.columns:
            features["breakeven_10y"] = panel["T10YIE"]
            features["breakeven_10y_change_20d"] = panel["T10YIE"].diff(20)

        # --- 政策利率 ---
        if "DFF" in panel.columns:
            features["fed_funds_rate"] = panel["DFF"]
            features["fed_funds_rate_change_60d"] = panel["DFF"].diff(60)

        # 实际联邦基金利率 (名义 - 通胀预期)
        if "DFF" in panel.columns and "T10YIE" in panel.columns:
            features["real_fed_funds"] = panel["DFF"] - panel["T10YIE"]

        # --- 财政因子 (中金核心因子) ---
        if "GFDEBTN" in panel.columns:
            debt = panel["GFDEBTN"]
            features["federal_debt"] = debt
            # 债务增速 (季度同比近似: 252个交易日)
            features["federal_debt_yoy"] = debt.pct_change(252)

        if "MTSDS133FMS" in panel.columns:
            deficit = panel["MTSDS133FMS"]
            features["fiscal_deficit"] = deficit
            # 赤字 12 个月滚动 (约 252 交易日)
            features["fiscal_deficit_12m_sum"] = deficit.rolling(252).sum()

        # --- 货币 ---
        if "M2SL" in panel.columns:
            m2 = panel["M2SL"]
            features["m2_yoy"] = m2.pct_change(252)

        # --- 美元 ---
        if "DTWEXBGS" in panel.columns:
            tw_usd = panel["DTWEXBGS"]
            features["tw_usd"] = tw_usd
            features["tw_usd_ret_20d"] = tw_usd.pct_change(20)
            features["tw_usd_zscore"] = (
                (tw_usd - tw_usd.rolling(252).mean()) /
                tw_usd.rolling(252).std().replace(0, np.nan)
            )

        # --- CPI ---
        if "CPIAUCSL" in panel.columns:
            cpi = panel["CPIAUCSL"]
            features["cpi_yoy"] = cpi.pct_change(252)

        return features

    def _add_cot_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """COT 持仓特征"""
        cot = self._load_cot()

        # 对齐到 GLD 交易日 (ffill, 周频 -> 日频)
        cot_aligned = cot.reindex(features.index, method="ffill")

        # 非商业净持仓 (投机头寸)
        if "NonComm_Net" in cot_aligned.columns:
            nc_net = cot_aligned["NonComm_Net"]
            features["cot_noncomm_net"] = nc_net
            # 持仓变化
            features["cot_noncomm_net_change"] = nc_net.diff(5)
            # 持仓历史分位数
            features["cot_noncomm_net_pctile"] = nc_net.rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False,
            )

        # 商业净持仓 (套保头寸)
        if "Comm_Net" in cot_aligned.columns:
            c_net = cot_aligned["Comm_Net"]
            features["cot_comm_net"] = c_net
            features["cot_comm_net_change"] = c_net.diff(5)

        # 总持仓量
        if "Open_Interest" in cot_aligned.columns:
            oi = cot_aligned["Open_Interest"]
            features["cot_open_interest"] = oi
            features["cot_oi_change_pct"] = oi.pct_change(5)

        return features

    def _add_cb_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """央行购金特征"""
        cb = self._load_cb_features()
        if cb.empty:
            return features

        # 对齐到 GLD 交易日 (ffill, 月频 -> 日频)
        cb_aligned = cb.reindex(features.index, method="ffill")

        for col in cb.columns:
            features[col] = cb_aligned[col]

        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    builder = MacroFeatureBuilder()
    df = builder.build()
    print(f"\n宏观面特征: {df.shape[1]} 列, {df.shape[0]} 行")
    print(f"列名: {list(df.columns)}")
    print(f"\n缺失率:")
    missing = df.isnull().mean()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
