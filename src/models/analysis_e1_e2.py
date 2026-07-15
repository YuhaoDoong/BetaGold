"""
E1: 区间校准分析 — 按 Regime / GVZ / 波动率 / 年度 分层检验覆盖率
E2: bp 分桶单调性 — Bull 下 bp 分6桶, 看 5d/10d/15d 胜率、均值、PF、MAE

用法:
    conda activate gold
    python src/models/analysis_e1_e2.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 数据加载
# ============================================================

def load_all():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))

    # Regime
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    # GVZ
    gvz_path = os.path.join(raw_dir, "volatility", "gvz.csv")
    gvz_df = pd.read_csv(gvz_path, index_col=0, parse_dates=True)
    gvz = gvz_df["GVZ"] if "GVZ" in gvz_df.columns else gvz_df.iloc[:, 0]

    # RV (from features)
    rv_20d = features["rv_20d"] if "rv_20d" in features.columns else None

    return gld, range_df, regime, gvz, rv_20d, features


# ============================================================
# E1: 区间校准
# ============================================================

def e1_interval_calibration(range_df, regime, gvz, rv_20d, gld):
    print("=" * 70)
    print("  E1: 区间校准分析")
    print("=" * 70)

    dates = range_df.index
    pred_u = range_df["pred_upper_pct"]
    pred_l = range_df["pred_lower_pct"]
    act_u = range_df["actual_upper_pct"]
    act_l = range_df["actual_lower_pct"]

    # 基本覆盖率
    upper_cov = (act_u <= pred_u)
    lower_cov = (act_l >= pred_l)
    both_cov = upper_cov & lower_cov
    pred_width = pred_u - pred_l
    act_width = act_u - act_l

    print(f"\n  === 总体 ===")
    print(f"  样本: {len(dates)} ({dates[0].strftime('%Y-%m')} ~ {dates[-1].strftime('%Y-%m')})")
    print(f"  覆盖率: {both_cov.mean():.1%} (上={upper_cov.mean():.1%}, 下={lower_cov.mean():.1%})")
    print(f"  预测宽度: {pred_width.mean():.2f}% (std={pred_width.std():.2f}%)")
    print(f"  实际宽度: {act_width.mean():.2f}% (std={act_width.std():.2f}%)")
    print(f"  宽度比: {(pred_width / act_width).mean():.2f}x")
    print(f"  紧凑度: {both_cov.mean() / pred_width.mean():.3f}")

    # --- 按 Regime ---
    reg_aligned = regime.reindex(dates)
    print(f"\n  === 按 Regime ===")
    print(f"  {'Regime':8s} {'N':>5s} {'Cov':>6s} {'U_cov':>6s} {'L_cov':>6s} "
          f"{'PredW':>6s} {'ActW':>6s} {'Ratio':>6s} {'Tight':>6s}")
    print(f"  {'-' * 56}")

    for reg in ["Bull", "Mixed", "Bear"]:
        mask = reg_aligned == reg
        if mask.sum() < 10:
            continue
        cov = both_cov[mask].mean()
        uc = upper_cov[mask].mean()
        lc = lower_cov[mask].mean()
        pw = pred_width[mask].mean()
        aw = act_width[mask].mean()
        ratio = pw / aw if aw > 0 else 0
        tight = cov / pw if pw > 0 else 0
        print(f"  {reg:8s} {mask.sum():5d} {cov:6.1%} {uc:6.1%} {lc:6.1%} "
              f"{pw:6.2f}% {aw:6.2f}% {ratio:6.2f}x {tight:6.3f}")

    # --- 按 GVZ 分位 ---
    gvz_aligned = gvz.reindex(dates)
    gvz_valid = gvz_aligned.dropna()
    if len(gvz_valid) > 100:
        print(f"\n  === 按 GVZ 分位 (n={len(gvz_valid)}) ===")
        gvz_q = pd.qcut(gvz_valid, 4, labels=["Q1低", "Q2", "Q3", "Q4高"])
        print(f"  {'GVZ分位':8s} {'N':>5s} {'GVZ范围':>14s} {'Cov':>6s} "
              f"{'PredW':>6s} {'ActW':>6s} {'Ratio':>6s}")
        print(f"  {'-' * 52}")

        for q in ["Q1低", "Q2", "Q3", "Q4高"]:
            mask = gvz_q == q
            idx = gvz_valid[mask].index
            cov = both_cov.reindex(idx).mean()
            pw = pred_width.reindex(idx).mean()
            aw = act_width.reindex(idx).mean()
            gvz_lo = gvz_valid[mask].min()
            gvz_hi = gvz_valid[mask].max()
            ratio = pw / aw if aw > 0 else 0
            print(f"  {q:8s} {mask.sum():5d} {gvz_lo:5.1f}~{gvz_hi:5.1f} "
                  f"{cov:6.1%} {pw:6.2f}% {aw:6.2f}% {ratio:6.2f}x")

    # --- 按 RV 分位 ---
    if rv_20d is not None:
        rv_aligned = rv_20d.reindex(dates).dropna()
        if len(rv_aligned) > 100:
            print(f"\n  === 按 RV20d 分位 (n={len(rv_aligned)}) ===")
            rv_q = pd.qcut(rv_aligned, 4, labels=["Q1低", "Q2", "Q3", "Q4高"])
            print(f"  {'RV分位':8s} {'N':>5s} {'RV范围':>14s} {'Cov':>6s} "
                  f"{'PredW':>6s} {'ActW':>6s} {'Ratio':>6s}")
            print(f"  {'-' * 52}")

            for q in ["Q1低", "Q2", "Q3", "Q4高"]:
                mask = rv_q == q
                idx = rv_aligned[mask].index
                cov = both_cov.reindex(idx).mean()
                pw = pred_width.reindex(idx).mean()
                aw = act_width.reindex(idx).mean()
                rv_lo = rv_aligned[mask].min()
                rv_hi = rv_aligned[mask].max()
                ratio = pw / aw if aw > 0 else 0
                print(f"  {q:8s} {mask.sum():5d} {rv_lo:5.3f}~{rv_hi:5.3f} "
                      f"{cov:6.1%} {pw:6.2f}% {aw:6.2f}% {ratio:6.2f}x")

    # --- 按年度 ---
    print(f"\n  === 按年度 ===")
    print(f"  {'年份':>6s} {'N':>5s} {'Cov':>6s} {'U_cov':>6s} {'L_cov':>6s} "
          f"{'PredW':>6s} {'ActW':>6s} {'Ratio':>6s} {'Regime分布':>20s}")
    print(f"  {'-' * 72}")

    years = pd.Series(dates.year, index=dates)
    for yr in sorted(years.unique()):
        mask = years == yr
        idx = dates[mask]
        cov = both_cov[mask].mean()
        uc = upper_cov[mask].mean()
        lc = lower_cov[mask].mean()
        pw = pred_width[mask].mean()
        aw = act_width[mask].mean()
        ratio = pw / aw if aw > 0 else 0

        # regime 分布
        reg_yr = reg_aligned[mask]
        bull_pct = (reg_yr == "Bull").mean()
        mixed_pct = (reg_yr == "Mixed").mean()
        bear_pct = (reg_yr == "Bear").mean()
        reg_str = f"B{bull_pct:.0%}/M{mixed_pct:.0%}/Br{bear_pct:.0%}"

        print(f"  {yr:6d} {mask.sum():5d} {cov:6.1%} {uc:6.1%} {lc:6.1%} "
              f"{pw:6.2f}% {aw:6.2f}% {ratio:6.2f}x {reg_str:>20s}")

    # --- 覆盖率校准图: 预测覆盖率 vs 实际 ---
    print(f"\n  === 覆盖率偏差分析 ===")
    # 上界突破幅度
    u_breach = (act_u - pred_u).clip(lower=0)  # 正 = 突破
    l_breach = (pred_l - act_l).clip(lower=0)  # 正 = 突破
    print(f"  上界突破: 发生率={( u_breach > 0).mean():.1%}, "
          f"均值={u_breach[u_breach > 0].mean():.2f}%, "
          f"max={u_breach.max():.2f}%")
    print(f"  下界突破: 发生率={(l_breach > 0).mean():.1%}, "
          f"均值={l_breach[l_breach > 0].mean():.2f}%, "
          f"max={l_breach.max():.2f}%")

    # 按 Regime 的突破
    print(f"\n  上界突破率 by Regime:")
    for reg in ["Bull", "Mixed", "Bear"]:
        mask = reg_aligned == reg
        if mask.sum() < 10:
            continue
        ub = (u_breach[mask] > 0).mean()
        lb = (l_breach[mask] > 0).mean()
        print(f"    {reg:8s}: 上界突破={ub:.1%}, 下界突破={lb:.1%}")


# ============================================================
# E2: bp 分桶单调性
# ============================================================

def build_band(range_df, gld_close):
    """Lag-avg band construction."""
    close = gld_close.reindex(range_df.index)
    uppers, lowers = [], []
    for lag in [1, 2, 3]:
        cl = close.shift(lag)
        pu = range_df["pred_upper_pct"].shift(lag)
        pl = range_df["pred_lower_pct"].shift(lag)
        uppers.append(cl * (1 + pu / 100))
        lowers.append(cl * (1 + pl / 100))
    upper_band = pd.concat(uppers, axis=1).mean(axis=1)
    lower_band = pd.concat(lowers, axis=1).mean(axis=1)
    bp = (close - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bp


def compute_forward_returns(gld, dates):
    """计算前瞻收益和 MAE (Maximum Adverse Excursion)."""
    close = gld["Close"]
    high = gld["High"]
    low = gld["Low"]

    result = pd.DataFrame(index=dates)

    for horizon in [5, 10, 15]:
        # 前瞻收益
        fwd_ret = (close.shift(-horizon) / close - 1) * 100
        result[f"fwd_ret_{horizon}d"] = fwd_ret.reindex(dates)

        # MAE: 持仓期间的最大不利波动 (对做多而言 = 最大回撤)
        mae = pd.Series(dtype=float, index=dates)
        for d in dates:
            loc = close.index.get_loc(d)
            end_loc = min(loc + horizon, len(close) - 1)
            if end_loc <= loc:
                continue
            future_lows = low.iloc[loc + 1: end_loc + 1]
            if len(future_lows) == 0:
                continue
            min_low = future_lows.min()
            entry_price = close.iloc[loc]
            mae.loc[d] = (min_low / entry_price - 1) * 100  # 负值 = 不利

        result[f"mae_{horizon}d"] = mae

        # MFE: Maximum Favorable Excursion (最大有利波动)
        mfe = pd.Series(dtype=float, index=dates)
        for d in dates:
            loc = close.index.get_loc(d)
            end_loc = min(loc + horizon, len(close) - 1)
            if end_loc <= loc:
                continue
            future_highs = high.iloc[loc + 1: end_loc + 1]
            if len(future_highs) == 0:
                continue
            max_high = future_highs.max()
            entry_price = close.iloc[loc]
            mfe.loc[d] = (max_high / entry_price - 1) * 100

        result[f"mfe_{horizon}d"] = mfe

    return result


def e2_bp_bucketing(range_df, gld, regime):
    print(f"\n{'=' * 70}")
    print("  E2: bp 分桶单调性分析")
    print("=" * 70)

    gld_close = gld["Close"]
    upper_band, lower_band, bp = build_band(range_df, gld_close)

    dates = bp.dropna().index
    reg_aligned = regime.reindex(dates)

    # 计算前瞻收益和 MAE
    print(f"\n  计算前瞻收益和 MAE/MFE... (n={len(dates)})")
    fwd = compute_forward_returns(gld, dates)

    # 合并数据
    df = pd.DataFrame({
        "bp": bp.reindex(dates),
        "regime": reg_aligned,
        "close": gld_close.reindex(dates),
    })
    df = df.join(fwd)
    df = df.dropna(subset=["bp", "fwd_ret_5d"])

    print(f"  有效样本: {len(df)}")

    # === 全样本 bp 分桶 ===
    bins = [-np.inf, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, np.inf]
    labels = ["<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.60", "0.60-0.80", ">0.80"]
    df["bp_bucket"] = pd.cut(df["bp"], bins=bins, labels=labels)

    print(f"\n  === 全样本 bp 分桶 ===")
    _print_bucket_table(df, labels)

    # === Bull Only bp 分桶 ===
    bull_df = df[df["regime"] == "Bull"].copy()
    print(f"\n  === Bull Only bp 分桶 (n={len(bull_df)}) ===")
    if len(bull_df) > 50:
        _print_bucket_table(bull_df, labels)

    # === Mixed bp 分桶 ===
    mixed_df = df[df["regime"] == "Mixed"].copy()
    print(f"\n  === Mixed bp 分桶 (n={len(mixed_df)}) ===")
    if len(mixed_df) > 50:
        _print_bucket_table(mixed_df, labels)

    # === Bear bp 分桶 ===
    bear_df = df[df["regime"] == "Bear"].copy()
    print(f"\n  === Bear bp 分桶 (n={len(bear_df)}) ===")
    if len(bear_df) > 20:
        _print_bucket_table(bear_df, labels)
    else:
        print(f"  样本太少 ({len(bear_df)}), 跳过")

    # === 单调性检验 ===
    print(f"\n  === 单调性检验 (Bull Only) ===")
    if len(bull_df) > 50:
        _monotonicity_test(bull_df, labels)

    # === 细粒度: Bull bp < 0.35 的逐 0.05 分析 ===
    print(f"\n  === Bull 低位细分 (bp 0~0.35, 每0.05) ===")
    if len(bull_df) > 50:
        fine_bins = [-np.inf, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, np.inf]
        fine_labels = ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20",
                       "0.20-0.25", "0.25-0.30", "0.30-0.35", ">0.35"]
        bull_df["bp_fine"] = pd.cut(bull_df["bp"], bins=fine_bins, labels=fine_labels)
        _print_bucket_table(bull_df, fine_labels, bucket_col="bp_fine")


def _print_bucket_table(df, labels, bucket_col="bp_bucket"):
    """打印分桶表格。"""
    print(f"  {'桶':>10s} {'N':>5s} "
          f"{'5d胜率':>7s} {'5d均值':>7s} {'5d中位':>7s} {'5dPF':>6s} "
          f"{'10d胜率':>7s} {'10d均值':>7s} "
          f"{'5dMAE':>7s} {'5dMFE':>7s} "
          f"{'10dMAE':>7s}")
    print(f"  {'-' * 90}")

    for label in labels:
        mask = df[bucket_col] == label
        n = mask.sum()
        if n < 5:
            print(f"  {label:>10s} {n:5d}  (样本不足)")
            continue

        sub = df[mask]

        # 5d
        wr5 = (sub["fwd_ret_5d"] > 0).mean()
        avg5 = sub["fwd_ret_5d"].mean()
        med5 = sub["fwd_ret_5d"].median()
        wins5 = sub.loc[sub["fwd_ret_5d"] > 0, "fwd_ret_5d"].sum()
        loss5 = abs(sub.loc[sub["fwd_ret_5d"] <= 0, "fwd_ret_5d"].sum())
        pf5 = wins5 / loss5 if loss5 > 0 else float("inf")

        # 10d
        wr10 = (sub["fwd_ret_10d"] > 0).mean() if "fwd_ret_10d" in sub else np.nan
        avg10 = sub["fwd_ret_10d"].mean() if "fwd_ret_10d" in sub else np.nan

        # MAE/MFE
        mae5 = sub["mae_5d"].mean() if "mae_5d" in sub else np.nan
        mfe5 = sub["mfe_5d"].mean() if "mfe_5d" in sub else np.nan
        mae10 = sub["mae_10d"].mean() if "mae_10d" in sub else np.nan

        print(f"  {label:>10s} {n:5d} "
              f"{wr5:7.1%} {avg5:+7.2f}% {med5:+7.2f}% {pf5:6.2f} "
              f"{wr10:7.1%} {avg10:+7.2f}% "
              f"{mae5:+7.2f}% {mfe5:+7.2f}% "
              f"{mae10:+7.2f}%")


def _monotonicity_test(df, labels, bucket_col="bp_bucket"):
    """检验 bp 越低, 收益是否越好 (单调性)。"""
    bucket_means = []
    for label in labels:
        mask = df[bucket_col] == label
        if mask.sum() < 5:
            bucket_means.append(np.nan)
        else:
            bucket_means.append(df.loc[mask, "fwd_ret_5d"].mean())

    # 检查前4个桶是否单调递减
    valid = [x for x in bucket_means[:4] if not np.isnan(x)]
    if len(valid) >= 3:
        decreasing = all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1))
        print(f"  5d均值 (低bp→高bp): {' → '.join(f'{x:+.2f}%' for x in valid)}")
        print(f"  单调递减: {'是' if decreasing else '否'}")

        # Spearman rank correlation: bp_bucket_midpoint vs fwd_ret
        from scipy import stats
        midpoints = [-0.05, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
        valid_pairs = [(m, r) for m, r in zip(midpoints, bucket_means) if not np.isnan(r)]
        if len(valid_pairs) >= 4:
            mids, rets = zip(*valid_pairs)
            rho, p = stats.spearmanr(mids, rets)
            print(f"  Spearman相关: rho={rho:.3f}, p={p:.3f}")
            if rho < -0.5 and p < 0.1:
                print(f"  → bp 与收益显著负相关 (bp越低收益越高)")
            elif rho < 0:
                print(f"  → bp 与收益弱负相关 (趋势正确但不强)")
            else:
                print(f"  → bp 与收益无负相关 (bp信号可能无效!)")

    # 同样检验 10d
    bucket_means_10 = []
    for label in labels:
        mask = df[bucket_col] == label
        if mask.sum() < 5:
            bucket_means_10.append(np.nan)
        else:
            bucket_means_10.append(df.loc[mask, "fwd_ret_10d"].mean())

    valid_10 = [x for x in bucket_means_10[:4] if not np.isnan(x)]
    if len(valid_10) >= 3:
        print(f"  10d均值 (低bp→高bp): {' → '.join(f'{x:+.2f}%' for x in valid_10)}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  E1 + E2: 区间校准 + bp 分桶分析")
    print("=" * 70)

    gld, range_df, regime, gvz, rv_20d, features = load_all()

    # E1
    e1_interval_calibration(range_df, regime, gvz, rv_20d, gld)

    # E2
    e2_bp_bucketing(range_df, gld, regime)

    print(f"\n{'=' * 70}")
    print("  分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
