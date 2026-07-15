"""
模型架构对比实验: LSTM vs Transformer vs ALSTM

在相同的数据拆分、超参、评估框架下对比三种模型架构在
5 日波动区间预测上的表现。

评估指标:
    - coverage: 实际价格落在预测区间内的比例
    - avg_width: 平均区间宽度 (越小越好，在覆盖率达标前提下)
    - tightness: coverage / avg_width (紧凑度, 越高越好)
    - val_loss: 验证集 quantile loss

用法:
    conda activate gold
    python src/models/train_dl_range_compare.py
"""

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.dl_fair_value import select_features
from src.models.dl_range_predictor import DLRangePredictor

warnings.filterwarnings("ignore")


def load_data():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    common = features.index.intersection(gld.index)
    return features.loc[common], gld.loc[common]


def build_targets(gld):
    close, high, low = gld["Close"], gld["High"], gld["Low"]
    max_high_5d = high.shift(-1).rolling(5).max().shift(-4)
    min_low_5d = low.shift(-1).rolling(5).min().shift(-4)
    upper_pct = (max_high_5d / close - 1) * 100
    lower_pct = (min_low_5d / close - 1) * 100
    log_ret = np.log(close / close.shift(1))
    rv_scale = log_ret.rolling(10).std() * np.sqrt(5) * 100
    return upper_pct, lower_pct, rv_scale


def eval_range(pred_upper, pred_lower, actual_upper, actual_lower):
    upper_covered = actual_upper <= pred_upper
    lower_covered = actual_lower >= pred_lower
    both = upper_covered & lower_covered
    pred_w = pred_upper - pred_lower
    actual_w = actual_upper - actual_lower
    avg_w = pred_w.mean()
    avg_aw = actual_w.mean()
    return {
        "coverage": both.mean(),
        "upper_cov": upper_covered.mean(),
        "lower_cov": lower_covered.mean(),
        "avg_width": avg_w,
        "avg_actual_width": avg_aw,
        "width_ratio": avg_w / avg_aw if avg_aw > 0 else 0,
        "tightness": both.mean() / avg_w if avg_w > 0 else 0,
    }


def run_model_walkforward(model_type, features, gld, feat_cols,
                          upper_s, lower_s, rv_s, dates,
                          q_upper=0.85, q_lower=0.15,
                          cal_target_cov=0.80):
    """对指定模型执行 walk-forward 评估。"""
    n = len(dates)
    min_train = 1260
    val_size = 252
    cal_size = 126
    test_size = 126
    step = 126
    seq_len = 20

    oos_pu = pd.Series(dtype=float)
    oos_pl = pd.Series(dtype=float)
    oos_au = pd.Series(dtype=float)
    oos_al = pd.Series(dtype=float)
    fold_results = []

    cutoff = min_train + cal_size
    fold_id = 0
    min_test = 60

    while cutoff + min_test <= n:
        fold_id += 1
        train_end = cutoff - cal_size
        cal_start = cutoff - cal_size
        test_end = min(cutoff + test_size, n)

        train_dates = dates[:train_end]
        val_start = max(0, train_end - val_size)
        val_dates = dates[val_start:train_end]
        cal_dates = dates[cal_start:cutoff]
        test_dates = dates[cutoff:test_end]

        if len(train_dates) < min_train:
            cutoff += step
            continue

        X_train = features.loc[train_dates, feat_cols].values
        u_train = upper_s.loc[train_dates].values
        l_train = lower_s.loc[train_dates].values
        rv_train = rv_s.loc[train_dates].values

        X_val = features.loc[val_dates, feat_cols].values
        u_val = upper_s.loc[val_dates].values
        l_val = lower_s.loc[val_dates].values
        rv_val = rv_s.loc[val_dates].values

        X_cal = features.loc[cal_dates, feat_cols].values
        u_cal = upper_s.loc[cal_dates].values
        l_cal = lower_s.loc[cal_dates].values
        rv_cal = rv_s.loc[cal_dates].values

        t0 = time.time()
        predictor = DLRangePredictor(
            seq_len=seq_len, hidden_size=64, num_layers=2,
            dropout=0.2, lr=1e-3, weight_decay=1e-4,
            epochs=150, batch_size=64, patience=20,
            q_upper=q_upper, q_lower=q_lower,
            n_ensemble=3, cal_target_cov=cal_target_cov,
            model_type=model_type,
        )
        predictor.fit(
            X_train, u_train, l_train, rv_train,
            X_val, u_val, l_val, rv_val,
            X_cal, u_cal, l_cal, rv_cal,
            verbose=False,
        )
        train_time = time.time() - t0

        # 预测
        n_prefix = seq_len - 1
        prefix_dates = dates[cutoff - n_prefix: cutoff]
        combined_dates = prefix_dates.append(test_dates)
        X_comb = features.loc[combined_dates, feat_cols].values
        rv_comb = rv_s.loc[combined_dates].values

        pred_u, pred_l = predictor.predict(X_comb, rv_comb)

        actual_u = upper_s.loc[test_dates].values
        actual_l = lower_s.loc[test_dates].values

        m = eval_range(pd.Series(pred_u), pd.Series(pred_l),
                       pd.Series(actual_u), pd.Series(actual_l))

        print(f"    F{fold_id:2d} "
              f"{test_dates[0].strftime('%y/%m')}~{test_dates[-1].strftime('%y/%m')} "
              f"cov={m['coverage']:.0%} w={m['avg_width']:.1f}% "
              f"tight={m['tightness']:.2f} "
              f"time={train_time:.0f}s")

        fold_results.append({
            "fold": fold_id,
            "test_start": test_dates[0], "test_end": test_dates[-1],
            "n_test": len(test_dates),
            "train_time": train_time,
            **m,
        })

        oos_pu = pd.concat([oos_pu, pd.Series(pred_u, index=test_dates)])
        oos_pl = pd.concat([oos_pl, pd.Series(pred_l, index=test_dates)])
        oos_au = pd.concat([oos_au, pd.Series(actual_u, index=test_dates)])
        oos_al = pd.concat([oos_al, pd.Series(actual_l, index=test_dates)])
        cutoff += step

    # 去重
    oos_pu = oos_pu[~oos_pu.index.duplicated(keep="last")]
    oos_pl = oos_pl[~oos_pl.index.duplicated(keep="last")]
    oos_au = oos_au[~oos_au.index.duplicated(keep="last")]
    oos_al = oos_al[~oos_al.index.duplicated(keep="last")]

    overall = eval_range(oos_pu, oos_pl, oos_au, oos_al)
    fold_df = pd.DataFrame(fold_results)

    return overall, fold_df, {
        "pred_upper": oos_pu, "pred_lower": oos_pl,
        "actual_upper": oos_au, "actual_lower": oos_al,
    }


def main():
    features, gld = load_data()
    feat_cols = select_features(features)
    upper_target, lower_target, rv_scale = build_targets(gld)

    valid = (features[feat_cols].notna().all(axis=1)
             & upper_target.notna() & lower_target.notna() & rv_scale.notna())
    feat_df = features.loc[valid]
    upper_s = upper_target[valid]
    lower_s = lower_target[valid]
    rv_s = rv_scale[valid]
    dates = feat_df.index.sort_values()

    print("=" * 70)
    print("  模型架构对比: LSTM vs Transformer vs ALSTM")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  特征={len(feat_cols)}, 样本={len(dates)}")
    print(f"  实际区间: w={(upper_s - lower_s).mean():.2f}%")
    print("=" * 70)

    model_types = ["lstm", "transformer", "alstm"]
    all_results = {}

    for mt in model_types:
        print(f"\n  {'='*60}")
        print(f"  模型: {mt.upper()}")
        print(f"  {'='*60}")
        t0 = time.time()
        overall, fold_df, oos = run_model_walkforward(
            mt, feat_df, gld, feat_cols, upper_s, lower_s, rv_s, dates)
        total_time = time.time() - t0
        all_results[mt] = {
            "overall": overall, "fold_df": fold_df,
            "oos": oos, "total_time": total_time,
        }

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n{'='*70}")
    print("  模型对比汇总")
    print(f"{'='*70}")

    print(f"\n  {'Model':14s} {'Cov':>5s} {'U_cov':>5s} {'L_cov':>5s} "
          f"{'Width':>6s} {'Ratio':>5s} {'Tight':>6s} "
          f"{'Cov_std':>7s} {'Time':>6s}")
    print(f"  {'-'*68}")

    best_name = None
    best_score = -1

    for name, res in all_results.items():
        o = res["overall"]
        fd = res["fold_df"]
        cov_std = fd["coverage"].std()
        avg_time = fd["train_time"].mean()
        score = o["tightness"] * (1 - cov_std)
        if score > best_score:
            best_score = score
            best_name = name
        print(f"  {name.upper():14s} {o['coverage']:5.0%} {o['upper_cov']:5.0%} "
              f"{o['lower_cov']:5.0%} {o['avg_width']:6.2f}% "
              f"{o['width_ratio']:5.1f}x {o['tightness']:6.3f} "
              f"{cov_std:7.1%} {avg_time:5.0f}s")

    print(f"\n  最佳模型: {best_name.upper()} "
          f"(score = tightness * stability = {best_score:.4f})")

    # 逐 fold 详细对比
    print(f"\n  逐 Fold 覆盖率对比:")
    print(f"  {'Fold':>4s}", end="")
    for mt in model_types:
        print(f"  {mt.upper():>12s}", end="")
    print()

    n_folds = min(len(all_results[mt]["fold_df"]) for mt in model_types)
    for i in range(n_folds):
        print(f"  F{i+1:2d} ", end="")
        for mt in model_types:
            fd = all_results[mt]["fold_df"]
            cov = fd.iloc[i]["coverage"]
            w = fd.iloc[i]["avg_width"]
            print(f"  {cov:5.0%} w={w:4.1f}%", end="")
        print()

    # 分波动率段对比
    print(f"\n  分波动率段对比:")
    first_oos = list(all_results.values())[0]["oos"]
    aw = first_oos["actual_upper"] - first_oos["actual_lower"]
    segments = [
        ("低波动 Q1", aw <= aw.quantile(0.25)),
        ("中波动 Q2-3", (aw > aw.quantile(0.25)) & (aw <= aw.quantile(0.75))),
        ("高波动 Q4", aw > aw.quantile(0.75)),
    ]

    for seg_name, mask in segments:
        print(f"\n  {seg_name} (n={mask.sum()}):")
        for mt in model_types:
            oos = all_results[mt]["oos"]
            s = eval_range(
                oos["pred_upper"][mask], oos["pred_lower"][mask],
                oos["actual_upper"][mask], oos["actual_lower"][mask])
            print(f"    {mt.upper():14s} cov={s['coverage']:.0%} "
                  f"w={s['avg_width']:.1f}% tight={s['tightness']:.3f}")

    # 保存最佳模型结果
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)

    best_oos = all_results[best_name]["oos"]
    result_df = pd.DataFrame({
        "pred_upper_pct": best_oos["pred_upper"],
        "pred_lower_pct": best_oos["pred_lower"],
        "actual_upper_pct": best_oos["actual_upper"],
        "actual_lower_pct": best_oos["actual_lower"],
        "gld_close": gld["Close"].reindex(best_oos["pred_upper"].index),
    })
    out_path = os.path.join(out_dir, "dl_range_v2_oos.parquet")
    result_df.to_parquet(out_path)
    print(f"\n  最佳模型 ({best_name.upper()}) 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
