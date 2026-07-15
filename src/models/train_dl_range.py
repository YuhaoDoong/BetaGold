"""
深度学习波动区间预测 Walk-Forward 评估 (V2: 归一化+独立校准+集成)

数据拆分 (每 fold):
    train: dates[:cutoff - cal_size]     (训练)
    val:   最后252天 of train            (early stopping)
    cal:   dates[cutoff-cal_size:cutoff]  (conformal calibration, 不参与训练)
    test:  dates[cutoff:cutoff+test_size] (OOS评估)

用法:
    conda activate gold
    python src/models/train_dl_range.py              # GLD (默认)
    python src/models/train_dl_range.py --asset slv  # 白银
"""

import argparse
import os
import sys
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


# asset -> (features parquet, raw csv, output suffix)
ASSET_CONFIG = {
    "gld": {
        "features": "features_all.parquet",
        "raw_csv": "gld.csv",
        "suffix": "v2",
    },
    "slv": {
        "features": "features_slv.parquet",
        "raw_csv": "slv.csv",
        "suffix": "slv",
    },
    # v3.7.105: GC=F 23h daily 训练 (替代 GLD 6.5h, 夜盘 info 保留 → Range/RV 更准)
    "gld_gc": {
        "features": "features_all.parquet",  # 宏观特征 reuse
        "raw_csv": "gold_futures.csv",         # GC=F 替代 gld.csv
        "suffix": "gc",
    },
}


def load_data(asset: str = "gld"):
    config = load_config()
    ac = ASSET_CONFIG[asset]
    proc = config["paths"]["processed_data"]
    feat_path = os.path.join(proc, ac["features"])
    features = pd.read_parquet(feat_path)

    raw_dir = config["paths"]["raw_data"]
    price = pd.read_csv(os.path.join(raw_dir, "market", ac["raw_csv"]),
                        index_col=0, parse_dates=True)
    common = features.index.intersection(price.index)
    return features.loc[common], price.loc[common]


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


def rv_based_range(gld, dates):
    close = gld["Close"]
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(20).std() * np.sqrt(252) * 100
    hw = (rv * np.sqrt(5) / np.sqrt(252) * 100).clip(1.5, 8.0)
    return hw.reindex(dates).values, -hw.reindex(dates).values


def run_config(features, gld, feat_cols, upper_s, lower_s, rv_s, dates,
               q_upper, q_lower, cal_target_cov, config_name):
    """单组参数的 walk-forward。"""
    n = len(dates)
    min_train = 1260
    val_size = 252
    cal_size = 126   # 独立校准集
    test_size = 126
    step = 126
    seq_len = 20

    oos_pu = pd.Series(dtype=float)
    oos_pl = pd.Series(dtype=float)
    oos_au = pd.Series(dtype=float)
    oos_al = pd.Series(dtype=float)
    fold_results = []

    cutoff = min_train + cal_size  # 确保 train + cal 都有数据
    fold_id = 0

    min_test = 60  # 允许最后一个fold用较短test期
    while cutoff + min_test <= n:
        fold_id += 1
        # 拆分
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

        predictor = DLRangePredictor(
            seq_len=seq_len, hidden_size=64, num_layers=2,
            dropout=0.2, lr=1e-3, weight_decay=1e-4,
            epochs=100, batch_size=64, patience=15,
            q_upper=q_upper, q_lower=q_lower,
            n_ensemble=2, cal_target_cov=cal_target_cov,
        )
        predictor.fit(
            X_train, u_train, l_train, rv_train,
            X_val, u_val, l_val, rv_val,
            X_cal, u_cal, l_cal, rv_cal,
            verbose=False,
        )

        # 预测
        n_prefix = seq_len - 1
        prefix_dates = dates[cutoff - n_prefix: cutoff]
        combined_dates = prefix_dates.append(test_dates)
        X_comb = features.loc[combined_dates, feat_cols].values
        rv_comb = rv_s.loc[combined_dates].values

        pred_u, pred_l = predictor.predict(X_comb, rv_comb)
        assert len(pred_u) == len(test_dates)

        actual_u = upper_s.loc[test_dates].values
        actual_l = lower_s.loc[test_dates].values

        m = eval_range(pd.Series(pred_u), pd.Series(pred_l),
                       pd.Series(actual_u), pd.Series(actual_l))

        print(f"    F{fold_id:2d} "
              f"{test_dates[0].strftime('%y/%m')}~{test_dates[-1].strftime('%y/%m')} "
              f"cov={m['coverage']:.0%} (u={m['upper_cov']:.0%} l={m['lower_cov']:.0%}) "
              f"w={m['avg_width']:.1f}% r={m['width_ratio']:.1f}x "
              f"cal_m=[+{predictor.cal_upper_margin:.1f}, +{predictor.cal_lower_margin:.1f}]")

        fold_results.append({
            "fold": fold_id,
            "test_start": test_dates[0], "test_end": test_dates[-1],
            "n_test": len(test_dates),
            "cal_upper": predictor.cal_upper_margin,
            "cal_lower": predictor.cal_lower_margin,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=list(ASSET_CONFIG.keys()),
                        default="gld")
    args = parser.parse_args()
    asset = args.asset
    suffix = ASSET_CONFIG[asset]["suffix"]

    features, price = load_data(asset)
    feat_cols = select_features(features)
    upper_target, lower_target, rv_scale = build_targets(price)

    valid = (features[feat_cols].notna().all(axis=1)
             & upper_target.notna() & lower_target.notna() & rv_scale.notna())
    feat_df = features.loc[valid]
    upper_s = upper_target[valid]
    lower_s = lower_target[valid]
    rv_s = rv_scale[valid]
    dates = feat_df.index.sort_values()

    print("=" * 70)
    print(f"  DL Range V2: 归一化 + 独立校准 + 集成  [asset={asset.upper()}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  特征={len(feat_cols)}, 样本={len(dates)}")
    print(f"  实际区间: w={( upper_s - lower_s).mean():.2f}%")
    print("=" * 70)

    # 周度再训练: 只跑已验证最优的 A 配置 (q85/15 cal80%).
    # 需要探索其他标定时, 在环境变量 DL_RANGE_FULL_GRID=1 下启用全量网格.
    if os.environ.get("DL_RANGE_FULL_GRID") == "1":
        configs = [
            {"name": "A: q85/15 cal80%",
             "q_upper": 0.85, "q_lower": 0.15, "cal_target_cov": 0.80},
            {"name": "B: q90/10 cal80%",
             "q_upper": 0.90, "q_lower": 0.10, "cal_target_cov": 0.80},
            {"name": "C: q85/15 cal70%",
             "q_upper": 0.85, "q_lower": 0.15, "cal_target_cov": 0.70},
        ]
    else:
        configs = [
            {"name": "A: q85/15 cal80%",
             "q_upper": 0.85, "q_lower": 0.15, "cal_target_cov": 0.80},
        ]

    all_results = {}
    for cfg in configs:
        print(f"\n  === {cfg['name']} ===")
        overall, fold_df, oos = run_config(
            feat_df, price, feat_cols, upper_s, lower_s, rv_s, dates,
            cfg["q_upper"], cfg["q_lower"], cfg["cal_target_cov"],
            cfg["name"])
        all_results[cfg["name"]] = {
            "overall": overall, "fold_df": fold_df, "oos": oos}

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("  汇总")
    print(f"{'='*70}")

    print(f"\n  {'Config':22s} {'Cov':>5s} {'U':>5s} {'L':>5s} "
          f"{'Width':>5s} {'Ratio':>5s} {'Tight':>5s} "
          f"{'Cov_std':>7s}")
    print(f"  {'-'*62}")

    best_name = None
    best_score = -1

    for name, res in all_results.items():
        o = res["overall"]
        cov_std = res["fold_df"]["coverage"].std()
        # 评分: 紧凑度 × 稳定性 (1 - cov_std)
        score = o["tightness"] * (1 - cov_std)
        if score > best_score:
            best_score = score
            best_name = name
        print(f"  {name:22s} {o['coverage']:5.0%} {o['upper_cov']:5.0%} "
              f"{o['lower_cov']:5.0%} {o['avg_width']:5.1f}% "
              f"{o['width_ratio']:5.1f}x {o['tightness']:5.2f} "
              f"{cov_std:7.1%}")

    # RV-based 对照
    first_oos = list(all_results.values())[0]["oos"]
    rv_u, rv_l = rv_based_range(price, first_oos["pred_upper"].index)
    rv_m = eval_range(
        pd.Series(rv_u, index=first_oos["pred_upper"].index),
        pd.Series(rv_l, index=first_oos["pred_upper"].index),
        first_oos["actual_upper"], first_oos["actual_lower"])
    print(f"  {'RV-based (对照)':22s} {rv_m['coverage']:5.0%} "
          f"{rv_m['upper_cov']:5.0%} {rv_m['lower_cov']:5.0%} "
          f"{rv_m['avg_width']:5.1f}% {rv_m['width_ratio']:5.1f}x "
          f"{rv_m['tightness']:5.2f}")

    print(f"\n  最佳: {best_name}")

    # 最佳配置详细分析
    best = all_results[best_name]
    fd = best["fold_df"]
    print(f"\n  覆盖率: mean={fd['coverage'].mean():.1%}, "
          f"std={fd['coverage'].std():.1%}, "
          f"range=[{fd['coverage'].min():.1%}, {fd['coverage'].max():.1%}]")
    print(f"  宽度:   mean={fd['avg_width'].mean():.2f}%, "
          f"std={fd['avg_width'].std():.2f}%")
    print(f"  校准 margin: upper mean={fd['cal_upper'].mean():+.2f}%, "
          f"lower mean={fd['cal_lower'].mean():+.2f}%")

    # 分段分析
    oos = best["oos"]
    aw = oos["actual_upper"] - oos["actual_lower"]
    print(f"\n  分段分析 ({best_name}):")
    for label, mask in [
        ("低波动 Q1", aw <= aw.quantile(0.25)),
        ("中波动 Q2-3", (aw > aw.quantile(0.25)) & (aw <= aw.quantile(0.75))),
        ("高波动 Q4", aw > aw.quantile(0.75)),
    ]:
        if mask.sum() < 10:
            continue
        seg = eval_range(
            oos["pred_upper"][mask], oos["pred_lower"][mask],
            oos["actual_upper"][mask], oos["actual_lower"][mask])
        print(f"    {label:12s} (n={mask.sum():4d}): "
              f"cov={seg['coverage']:.0%}, w={seg['avg_width']:.1f}%, "
              f"actual={seg['avg_actual_width']:.1f}%, "
              f"ratio={seg['width_ratio']:.1f}x")

    # 保存
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    result_df = pd.DataFrame({
        "pred_upper_pct": oos["pred_upper"],
        "pred_lower_pct": oos["pred_lower"],
        "actual_upper_pct": oos["actual_upper"],
        "actual_lower_pct": oos["actual_lower"],
        "gld_close": price["Close"].reindex(oos["pred_upper"].index),
    })
    out_path = os.path.join(out_dir, f"dl_range_{suffix}_oos.parquet")
    result_df.to_parquet(out_path)
    print(f"\n  保存 OOS: {out_path}")

    # ============================================================
    # 训练最终生产模型 (全量数据, 用于 live inference)
    # ============================================================
    print(f"\n{'='*70}")
    print("  训练最终生产模型 (全量数据)")
    print(f"{'='*70}")

    best_cfg = next(c for c in configs if c["name"] == best_name)

    cal_size = 126
    val_size = 252
    n_total = len(dates)
    train_end = n_total - cal_size
    val_start = max(0, train_end - val_size)

    train_dates_final = dates[:train_end]
    val_dates_final = dates[val_start:train_end]
    cal_dates_final = dates[train_end:]

    X_tr = feat_df.loc[train_dates_final, feat_cols].values
    u_tr = upper_s.loc[train_dates_final].values
    l_tr = lower_s.loc[train_dates_final].values
    rv_tr = rv_s.loc[train_dates_final].values

    X_va = feat_df.loc[val_dates_final, feat_cols].values
    u_va = upper_s.loc[val_dates_final].values
    l_va = lower_s.loc[val_dates_final].values
    rv_va = rv_s.loc[val_dates_final].values

    X_ca = feat_df.loc[cal_dates_final, feat_cols].values
    u_ca = upper_s.loc[cal_dates_final].values
    l_ca = lower_s.loc[cal_dates_final].values
    rv_ca = rv_s.loc[cal_dates_final].values

    final_predictor = DLRangePredictor(
        seq_len=20, hidden_size=64, num_layers=2,
        dropout=0.2, lr=1e-3, weight_decay=1e-4,
        epochs=100, batch_size=64, patience=15,
        q_upper=best_cfg["q_upper"], q_lower=best_cfg["q_lower"],
        n_ensemble=2, cal_target_cov=best_cfg["cal_target_cov"],
    )
    final_predictor.fit(
        X_tr, u_tr, l_tr, rv_tr,
        X_va, u_va, l_va, rv_va,
        X_ca, u_ca, l_ca, rv_ca,
        verbose=False,
    )

    model_path = os.path.join(out_dir, f"dl_range_{suffix}_model.pkl")
    final_predictor.save(model_path)
    print(f"  保存生产模型: {model_path}")
    print(f"  特征数: {final_predictor.n_features}, "
          f"模型数: {len(final_predictor.models)}")
    print(f"  校准 margin: upper=+{final_predictor.cal_upper_margin:.2f}%, "
          f"lower=+{final_predictor.cal_lower_margin:.2f}%")

    # 保存特征列名 (供 live inference 精确对齐)
    feat_cols_path = os.path.join(out_dir, f"dl_range_{suffix}_features.txt")
    with open(feat_cols_path, "w") as f:
        f.write("\n".join(feat_cols))
    print(f"  保存特征列: {feat_cols_path}")


if __name__ == "__main__":
    main()
