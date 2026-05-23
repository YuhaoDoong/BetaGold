"""v3.7.223: Walk-forward validation — 判断生产参数是否过拟合.

方法:
  滚动 train/test split:
    fold 0: train 2017-2021, test 2022
    fold 1: train 2018-2022, test 2023
    fold 2: train 2019-2023, test 2024
    fold 3: train 2020-2024, test 2025
    fold 4: train 2021-2025, test 2026 (YTD)

  对每个 fold:
    1. 在 train 窗口跑 grid (rv cutoffs / ret cutoffs / bp_low cutoffs)
    2. 按 scoreB 选 train 上的最优 cutoff
    3. 把该 cutoff 应用到 test 窗口, 报告 test OOS scoreB / WR / sum

判断标准:
  - **robust**: 多 fold 选出同样 / 接近的 cutoff, OOS WR ≈ train WR
  - **overfit**: fold 之间选出不同 cutoff, OOS WR << train WR

测试参数:
  rv_pctile_max_hard (当前 prod=0.75)
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier


def build_signal_universe():
    """跑一遍 generate_daily_signals 拿全 BUY 信号 + 前向回报."""
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]; low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz["Close"])
    buy = sig[sig["buy_signal"]].copy()
    ret20 = close.pct_change(20)
    buy["ret_20d"] = ret20.reindex(buy.index) * 100
    # 前向回报 5/10/20d
    for h in (5, 10, 20):
        for d in buy.index:
            if d not in ohlc.index: continue
            i = ohlc.index.get_loc(d)
            ent = float(ohlc.iloc[i]["Open"])
            if i + h < len(ohlc):
                ext = float(ohlc.iloc[i+h]["Close"])
                buy.loc[d, f"r{h}d"] = (ext / ent - 1) * 100
            else:
                buy.loc[d, f"r{h}d"] = np.nan
    return buy


def score(sub: pd.DataFrame, horizon: int = 5) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "WR": None, "mean": None, "sum": None, "scoreB": 0}
    s = sub[f"r{horizon}d"].dropna()
    if not len(s):
        return {"n": n, "WR": None, "mean": None, "sum": None, "scoreB": 0}
    wr = (s > 0).mean()
    mean = s.mean()
    sumv = s.sum()
    scoreB = (wr ** 2) * math.log(1 + n) * mean
    return {"n": n, "WR": round(wr * 100, 1), "mean": round(mean, 2),
             "sum": round(sumv, 1), "scoreB": round(scoreB, 2),
             "max_loss": round(s.min(), 1)}


def grid_search_train(buy: pd.DataFrame, mask_train: pd.Series,
                        cutoffs: list, factor_col: str,
                        compare: str = "<") -> tuple:
    """在 train 窗口跑 cutoff grid, 返回 (best_cutoff, best_score, all_results)."""
    sub_train = buy[mask_train]
    results = []
    for c in cutoffs:
        if compare == "<":
            m = sub_train[factor_col] < c
        elif compare == "<=":
            m = sub_train[factor_col] <= c
        elif compare == ">":
            m = sub_train[factor_col] > c
        elif compare == ">=":
            m = sub_train[factor_col] >= c
        s = score(sub_train[m])
        results.append({"cutoff": c, **s})
    # 选 scoreB 最大 (n>=10 保证统计有效)
    valid = [r for r in results if r["n"] >= 10]
    if not valid:
        return None, None, results
    best = max(valid, key=lambda r: r["scoreB"])
    return best["cutoff"], best, results


def apply_test(buy: pd.DataFrame, mask_test: pd.Series,
                 cutoff: float, factor_col: str, compare: str = "<") -> dict:
    sub_test = buy[mask_test]
    if compare == "<": m = sub_test[factor_col] < cutoff
    elif compare == "<=": m = sub_test[factor_col] <= cutoff
    elif compare == ">": m = sub_test[factor_col] > cutoff
    elif compare == ">=": m = sub_test[factor_col] >= cutoff
    return score(sub_test[m])


def walk_forward(buy: pd.DataFrame, factor_col: str, compare: str,
                    cutoffs: list, train_years: int = 4):
    """Walk-forward: train_years 训练 / 1 年 OOS test, 滚动."""
    if buy.index.tz is not None:
        buy = buy.copy(); buy.index = buy.index.tz_localize(None)
    years_with_data = sorted(set(buy.index.year))
    if len(years_with_data) < train_years + 1:
        print(f"⚠️ 数据年份不足 ({len(years_with_data)} < {train_years+1})")
        return
    folds = []
    for test_year in range(years_with_data[train_years], years_with_data[-1] + 1):
        train_start = test_year - train_years
        train_mask = (buy.index.year >= train_start) & (buy.index.year < test_year)
        test_mask = buy.index.year == test_year
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue
        best_cutoff, best_train, all_train = grid_search_train(
            buy, train_mask, cutoffs, factor_col, compare)
        if best_cutoff is None:
            continue
        test_perf = apply_test(buy, test_mask, best_cutoff, factor_col, compare)
        # baseline (test 上不过滤)
        test_baseline = score(buy[test_mask])
        folds.append({
            "test_year": test_year,
            "train_period": f"{train_start}-{test_year-1}",
            "best_cutoff": best_cutoff,
            "train_n": best_train["n"], "train_WR": best_train["WR"],
            "train_sum": best_train["sum"], "train_scoreB": best_train["scoreB"],
            "test_n": test_perf["n"], "test_WR": test_perf["WR"],
            "test_sum": test_perf["sum"], "test_scoreB": test_perf["scoreB"],
            "test_max_loss": test_perf["max_loss"],
            "baseline_n": test_baseline["n"], "baseline_WR": test_baseline["WR"],
            "baseline_sum": test_baseline["sum"], "baseline_scoreB": test_baseline["scoreB"],
        })
    return folds


def main():
    print("加载 BUY 信号 + 前向回报 ...")
    buy = build_signal_universe()
    print(f"BUY 信号总数: {len(buy)} ({buy.index.min().date()} → {buy.index.max().date()})")
    print()

    print("=" * 100)
    print("【1】rv_pctile_max_hard walk-forward (train 4y → test 1y)")
    print("=" * 100)
    rv_cutoffs = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
    rv_folds = walk_forward(buy, "rv_pctile", "<", rv_cutoffs, train_years=4)
    df_rv = pd.DataFrame(rv_folds)
    print(df_rv.to_string(index=False))
    print()
    print(f"  best_cutoff 跨 fold 分布: {df_rv['best_cutoff'].tolist()}")
    print(f"  test_scoreB 平均: {df_rv['test_scoreB'].mean():.2f}  vs baseline 平均: {df_rv['baseline_scoreB'].mean():.2f}")
    print(f"  test_WR 平均: {df_rv['test_WR'].mean():.1f}%  vs baseline {df_rv['baseline_WR'].mean():.1f}%")

    print()
    print("=" * 100)
    print("【2】ret_20d_max_hard walk-forward — 看 ret_20d>X 撤底拦顶")
    print("=" * 100)
    ret_cutoffs = [-10, -5, -3, -1, 0, +1, +3, +5]
    ret_folds = walk_forward(buy, "ret_20d", ">", ret_cutoffs, train_years=4)
    df_ret = pd.DataFrame(ret_folds)
    print(df_ret.to_string(index=False))
    print()
    print(f"  best_cutoff 跨 fold 分布: {df_ret['best_cutoff'].tolist()}")
    print(f"  test_scoreB 平均: {df_ret['test_scoreB'].mean():.2f}  vs baseline {df_ret['baseline_scoreB'].mean():.2f}")

    print()
    print("=" * 100)
    print("【3】sp_score_threshold walk-forward (这个是 buy_type SP↔BC 决策切点)")
    print("=" * 100)
    sp_cutoffs = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # sp_score 需要重新 join 进 buy — 已经在 sig 里, 但 generate_daily_signals 已包含
    if "sp_score" in buy.columns:
        sp_folds = walk_forward(buy, "sp_score", ">=", sp_cutoffs, train_years=4)
        df_sp = pd.DataFrame(sp_folds)
        print(df_sp.to_string(index=False))
        print(f"  best_cutoff 跨 fold 分布: {df_sp['best_cutoff'].tolist()}")
    else:
        print("⚠️ buy 表里没 sp_score 列")

    out = "/Users/yhdong/Gold/data/backtest_history/walk_forward_validate.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([
        df_rv.assign(factor="rv_pctile"),
        df_ret.assign(factor="ret_20d"),
    ], ignore_index=True).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
