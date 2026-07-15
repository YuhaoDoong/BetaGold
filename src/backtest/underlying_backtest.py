"""
Phase 4A: 标的预测回测

验证 Phase 3 OOS 预测信号是否有稳定的信息含量和经济价值。
不涉及期权结构，只看 GLD 标的层面的方向/波动信号。

运行：
    python src/backtest/underlying_backtest.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from scipy import stats

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 数据加载
# ============================================================

def load_gld_prices(config: dict) -> pd.Series:
    """加载 GLD 收盘价序列"""
    path = os.path.join(config["paths"]["raw_data"], "market", "gld.csv")
    gld = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return gld["Close"].sort_index()


def load_predictions() -> pd.DataFrame:
    """加载 Phase 3 OOS 预测"""
    path = os.path.join(ROOT, "data", "models", "baseline_predictions.parquet")
    pred = pd.read_parquet(path)
    pred["date"] = pd.to_datetime(pred["date"])
    return pred


def load_features_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载特征和标签（用于 benchmark 计算）"""
    config = load_config()
    proc = config["paths"]["processed_data"]
    features = pd.read_parquet(os.path.join(proc, "features_all.parquet"))
    labels = pd.read_parquet(os.path.join(proc, "labels.parquet"))
    return features, labels


# ============================================================
# 策略构造
# ============================================================

def build_signal_df(pred: pd.DataFrame, target: str, model: str,
                    gld_close: pd.Series) -> pd.DataFrame:
    """
    构建信号 DataFrame:
        - date, y_pred, y_true, gld_close, fwd_ret (实际未来收益)
    """
    sub = pred[(pred["target"] == target) & (pred["model"] == model)].copy()
    sub = sub.set_index("date").sort_index()

    df = pd.DataFrame(index=sub.index)
    df["y_pred"] = sub["y_pred"]
    df["y_true"] = sub["y_true"]
    df["fold"] = sub["fold"]
    df["gld_close"] = gld_close.reindex(df.index)

    # 计算实际 1 日收益（用于逐日 PnL 计算）
    df["daily_ret"] = gld_close.pct_change().reindex(df.index)

    return df.dropna(subset=["gld_close", "daily_ret"])


def strategy_direction(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """
    策略 A1: 纯方向信号
    - y_pred > 0: 做多 (position=1)
    - y_pred <= 0: 空仓 (position=0)

    持仓按日更新（预测每日更新，日频调仓）
    """
    return (df["y_pred"] > 0).astype(float)


def strategy_direction_vol_filter(df: pd.DataFrame,
                                  pred_all: pd.DataFrame,
                                  vol_target: str = "fwd_rv_20d",
                                  vol_model: str = "Ridge",
                                  vol_pctile: float = 0.80) -> pd.Series:
    """
    策略 A2: 方向 + 波动率过滤
    - 方向信号成立（y_pred > 0）
    - 且波动率预测不超过历史 80 分位
    """
    # 获取波动率预测
    vol_sub = pred_all[(pred_all["target"] == vol_target) &
                       (pred_all["model"] == vol_model)].copy()
    vol_sub = vol_sub.set_index(pd.to_datetime(vol_sub["date"]))["y_pred"]
    vol_sub = vol_sub.rename("vol_pred")

    # 计算滚动分位数 (expanding, OOS safe)
    vol_expanding_q = vol_sub.expanding(min_periods=60).quantile(vol_pctile)

    # 合并
    pos = (df["y_pred"] > 0).astype(float)

    # 波动率过滤：预测值超过 expanding 分位数时不入场
    vol_pred_aligned = vol_sub.reindex(df.index)
    vol_q_aligned = vol_expanding_q.reindex(df.index)
    vol_filter = vol_pred_aligned <= vol_q_aligned

    return pos * vol_filter.astype(float)


def strategy_regime_filter(df: pd.DataFrame,
                           features: pd.DataFrame) -> pd.Series:
    """
    策略 A3: Regime 过滤
    - 用 GVZ 分位数 (gvz_pctile_252d) 判断 regime
    - gvz_high=1 (高波 regime) 时缩减仓位到 50%
    """
    pos = (df["y_pred"] > 0).astype(float)

    if "gvz_high" in features.columns:
        gvz_high = features["gvz_high"].reindex(df.index)
        # 高波时仓位减半
        pos = pos * gvz_high.map({1: 0.5, 0: 1.0}).fillna(1.0)

    return pos


def benchmark_buy_hold(df: pd.DataFrame) -> pd.Series:
    """Benchmark: Buy & Hold GLD"""
    return pd.Series(1.0, index=df.index)


def benchmark_ma_crossover(df: pd.DataFrame,
                           gld_close: pd.Series,
                           fast: int = 20, slow: int = 60) -> pd.Series:
    """Benchmark: MA 交叉趋势策略"""
    ma_fast = gld_close.rolling(fast).mean()
    ma_slow = gld_close.rolling(slow).mean()
    signal = (ma_fast > ma_slow).astype(float)
    return signal.reindex(df.index).fillna(0.0)


# ============================================================
# 性能评估
# ============================================================

def calc_performance(daily_ret: pd.Series, positions: pd.Series,
                     name: str, cost_per_trade: float = 0.0001) -> dict:
    """
    计算策略绩效指标

    Parameters:
        daily_ret: GLD 每日收益率
        positions: 每日仓位 (0~1)
        name: 策略名
        cost_per_trade: 每次调仓成本 (单边, 占资产比)
    """
    # 策略收益
    strat_ret = daily_ret * positions

    # 交易成本：每次仓位变化扣成本
    pos_change = positions.diff().abs().fillna(0)
    costs = pos_change * cost_per_trade
    strat_ret_net = strat_ret - costs

    # 累计收益
    cum_ret = (1 + strat_ret_net).cumprod()

    # 年化
    n_days = len(strat_ret_net)
    n_years = n_days / 252
    total_ret = cum_ret.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe
    ann_vol = strat_ret_net.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = strat_ret_net[strat_ret_net < 0].std() * np.sqrt(252)
    sortino = cagr / downside if downside > 0 else 0

    # Max Drawdown
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()

    # Win rate (daily)
    active_days = strat_ret_net[positions > 0]
    win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0

    # Profit factor
    gains = active_days[active_days > 0].sum()
    losses = abs(active_days[active_days < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    # Turnover
    avg_turnover = pos_change.mean() * 252  # 年化换手

    return {
        "name": name,
        "total_ret": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_turnover": avg_turnover,
        "n_days": n_days,
        "exposure": positions.mean(),  # 平均仓位
    }


def calc_prediction_quality(df: pd.DataFrame, target: str) -> dict:
    """
    计算预测质量指标
    """
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    # IC (Spearman)
    ic, ic_pval = stats.spearmanr(y_pred, y_true)

    # 分组分析 (5 组)
    df_temp = df[["y_pred", "y_true"]].copy()
    df_temp["quintile"] = pd.qcut(df_temp["y_pred"], 5, labels=False,
                                  duplicates="drop")
    group_means = df_temp.groupby("quintile")["y_true"].mean()
    monotonicity = group_means.corr(pd.Series(range(len(group_means))))

    # Top-Bottom spread
    if len(group_means) >= 2:
        tb_spread = group_means.iloc[-1] - group_means.iloc[0]
    else:
        tb_spread = 0

    # 分 fold IC
    fold_ics = []
    for fold_id in sorted(df["fold"].unique()):
        fold_data = df[df["fold"] == fold_id]
        if len(fold_data) > 30:
            fold_ic, _ = stats.spearmanr(fold_data["y_pred"], fold_data["y_true"])
            fold_ics.append(fold_ic)
    ic_stability = np.mean([x > 0 for x in fold_ics]) if fold_ics else 0

    return {
        "target": target,
        "ic": ic,
        "ic_pval": ic_pval,
        "tb_spread": tb_spread,
        "monotonicity": monotonicity,
        "ic_positive_folds": ic_stability,
        "n_folds": len(fold_ics),
        "fold_ics": fold_ics,
    }


def calc_annual_performance(daily_ret: pd.Series, positions: pd.Series,
                            cost_per_trade: float = 0.0001) -> pd.DataFrame:
    """分年度计算绩效"""
    strat_ret = daily_ret * positions
    pos_change = positions.diff().abs().fillna(0)
    costs = pos_change * cost_per_trade
    strat_ret_net = strat_ret - costs

    years = strat_ret_net.index.year
    records = []
    for y in sorted(years.unique()):
        yr_ret = strat_ret_net[years == y]
        cum = (1 + yr_ret).cumprod()
        total = cum.iloc[-1] - 1
        vol = yr_ret.std() * np.sqrt(252)
        sharpe = (total / (len(yr_ret) / 252)) / vol if vol > 0 else 0
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min()
        records.append({
            "year": y,
            "return": total,
            "volatility": vol,
            "sharpe": sharpe,
            "max_dd": dd,
            "n_days": len(yr_ret),
        })
    return pd.DataFrame(records)


# ============================================================
# 分组收益分析
# ============================================================

def quintile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    5 分组收益分析
    按预测值分 5 组，计算每组的平均实际收益
    """
    df_temp = df[["y_pred", "y_true"]].dropna().copy()
    df_temp["quintile"] = pd.qcut(df_temp["y_pred"], 5,
                                  labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"],
                                  duplicates="drop")
    result = df_temp.groupby("quintile", observed=True).agg(
        mean_pred=("y_pred", "mean"),
        mean_actual=("y_true", "mean"),
        std_actual=("y_true", "std"),
        count=("y_true", "count"),
    )
    return result


# ============================================================
# 主入口
# ============================================================

def run_backtest():
    """运行完整的 Phase 4A 标的层回测"""
    config = load_config()

    logger.info("=" * 60)
    logger.info("Phase 4A: 标的预测回测")
    logger.info("=" * 60)

    # 加载数据
    gld_close = load_gld_prices(config)
    pred = load_predictions()
    features, labels = load_features_labels()

    logger.info(f"GLD 价格: {len(gld_close)} 行, "
                f"{gld_close.index.min().date()} ~ {gld_close.index.max().date()}")
    logger.info(f"OOS 预测: {len(pred)} 行")

    # 回测的目标和模型组合
    backtest_configs = [
        ("fwd_ret_10d", "Ridge"),
        ("fwd_ret_10d", "XGBoost"),
        ("fwd_ret_20d", "Ridge"),
        ("fwd_ret_20d", "XGBoost"),
    ]

    all_perf = []
    all_quality = []
    all_annual = {}
    all_quintile = {}

    for target, model in backtest_configs:
        logger.info(f"\n--- {target} / {model} ---")

        # 构建信号
        df = build_signal_df(pred, target, model, gld_close)
        if len(df) == 0:
            logger.warning(f"  无数据, 跳过")
            continue

        logger.info(f"  信号期间: {df.index.min().date()} ~ {df.index.max().date()}, "
                    f"{len(df)} 天")

        # 预测质量
        quality = calc_prediction_quality(df, f"{target}/{model}")
        all_quality.append(quality)
        logger.info(f"  IC={quality['ic']:.4f} (p={quality['ic_pval']:.4f}), "
                    f"TB_spread={quality['tb_spread']:.6f}, "
                    f"IC正折占比={quality['ic_positive_folds']:.0%}")

        # 分组分析
        q_result = quintile_analysis(df)
        all_quintile[f"{target}/{model}"] = q_result

        # === 策略回测 ===

        # A1: 纯方向
        pos_a1 = strategy_direction(df)
        perf_a1 = calc_performance(df["daily_ret"], pos_a1,
                                   f"A1_方向/{target}/{model}")
        all_perf.append(perf_a1)

        # A2: 方向 + 波动率过滤
        pos_a2 = strategy_direction_vol_filter(df, pred)
        perf_a2 = calc_performance(df["daily_ret"], pos_a2,
                                   f"A2_方向+Vol/{target}/{model}")
        all_perf.append(perf_a2)

        # A3: Regime 过滤
        pos_a3 = strategy_regime_filter(df, features)
        perf_a3 = calc_performance(df["daily_ret"], pos_a3,
                                   f"A3_Regime/{target}/{model}")
        all_perf.append(perf_a3)

        # Benchmark: Buy & Hold
        pos_bh = benchmark_buy_hold(df)
        perf_bh = calc_performance(df["daily_ret"], pos_bh, "BM_BuyHold")

        # Benchmark: MA 20/60
        pos_ma = benchmark_ma_crossover(df, gld_close)
        perf_ma = calc_performance(df["daily_ret"], pos_ma, "BM_MA20_60")

        # 只加一次 benchmark
        if target == backtest_configs[0][0] and model == backtest_configs[0][1]:
            all_perf.append(perf_bh)
            all_perf.append(perf_ma)

        # 年度表现 (用 A1)
        annual = calc_annual_performance(df["daily_ret"], pos_a1)
        all_annual[f"{target}/{model}"] = annual

        # 打印摘要
        for p in [perf_a1, perf_a2, perf_a3]:
            logger.info(f"  {p['name']}: CAGR={p['cagr']:.2%}, "
                        f"Sharpe={p['sharpe']:.2f}, MaxDD={p['max_dd']:.2%}, "
                        f"WinRate={p['win_rate']:.1%}, Exposure={p['exposure']:.1%}")

    # === 汇总结果 ===
    logger.info("\n" + "=" * 60)
    logger.info("汇总结果")
    logger.info("=" * 60)

    # 性能总表
    perf_df = pd.DataFrame(all_perf)
    cols_order = ["name", "cagr", "sharpe", "sortino", "max_dd",
                  "win_rate", "profit_factor", "exposure", "total_ret"]
    perf_df = perf_df[cols_order]

    logger.info("\n--- 绩效总表 ---")
    for _, row in perf_df.iterrows():
        logger.info(f"  {row['name']:40s} CAGR={row['cagr']:+.2%}  "
                    f"Sharpe={row['sharpe']:.2f}  MaxDD={row['max_dd']:.2%}  "
                    f"Exposure={row['exposure']:.0%}")

    # 预测质量总表
    logger.info("\n--- 预测质量 ---")
    for q in all_quality:
        logger.info(f"  {q['target']:30s} IC={q['ic']:.4f}  "
                    f"TB_spread={q['tb_spread']:.6f}  "
                    f"IC+folds={q['ic_positive_folds']:.0%}")

    # 分组收益
    logger.info("\n--- 分组收益 (5分位) ---")
    for key, q_df in all_quintile.items():
        logger.info(f"\n  {key}:")
        for idx, row in q_df.iterrows():
            logger.info(f"    {idx}: 预测均值={row['mean_pred']:.6f}, "
                        f"实际均值={row['mean_actual']:.6f} (n={row['count']:.0f})")

    # === 保存结果 ===
    out_dir = os.path.join(ROOT, "data", "backtest")
    os.makedirs(out_dir, exist_ok=True)

    perf_df.to_parquet(os.path.join(out_dir, "phase4a_performance.parquet"))

    quality_df = pd.DataFrame([{k: v for k, v in q.items() if k != "fold_ics"}
                                for q in all_quality])
    quality_df.to_parquet(os.path.join(out_dir, "phase4a_prediction_quality.parquet"))

    # 保存分组收益
    quintile_records = []
    for key, q_df in all_quintile.items():
        q_df_out = q_df.copy()
        q_df_out["config"] = key
        quintile_records.append(q_df_out)
    if quintile_records:
        pd.concat(quintile_records).to_parquet(
            os.path.join(out_dir, "phase4a_quintile.parquet"))

    # 保存年度表现
    annual_records = []
    for key, a_df in all_annual.items():
        a_out = a_df.copy()
        a_out["config"] = key
        annual_records.append(a_out)
    if annual_records:
        pd.concat(annual_records).to_parquet(
            os.path.join(out_dir, "phase4a_annual.parquet"))

    logger.info(f"\n结果保存到 {out_dir}/")
    logger.info("Phase 4A 完成")

    return perf_df, all_quality, all_quintile, all_annual


if __name__ == "__main__":
    run_backtest()
