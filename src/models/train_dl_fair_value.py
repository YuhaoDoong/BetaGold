"""
深度学习公允价格 Walk-Forward 评估 (LSTM vs Transformer)

Walk-forward 设置:
- 最小训练: 1260天 (5年)
- 验证: 最后 252天 (训练集末尾)
- 测试: 126天 (半年)
- 步进: 126天 (半年推进)
- 每个 fold 重新训练模型

支持模型:
- lstm: LSTM + Attention
- transformer: Transformer Encoder + Mean Pool

用法:
    conda activate gold
    python src/models/train_dl_fair_value.py               # 默认跑两个模型对比
    python src/models/train_dl_fair_value.py --model lstm
    python src/models/train_dl_fair_value.py --model transformer
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.dl_fair_value import (
    DLFairValuePredictor, select_features,
)

warnings.filterwarnings("ignore")


def load_data():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    gld_close = gld["Close"].rename("gld_close")

    common = features.index.intersection(gld_close.index)
    features, gld_close = features.loc[common], gld_close.loc[common]
    return features, gld_close


def run_walk_forward(model_type: str = "lstm"):
    features, gld_close = load_data()

    model_name = {"lstm": "LSTM+Attention",
                  "transformer": "Transformer"}[model_type]
    print("=" * 70)
    print(f"  Phase 4A DL: {model_name} 公允价格预测 (Walk-Forward)")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 选特征
    feat_cols = select_features(features)
    print(f"  选用特征: {len(feat_cols)} 个")

    # 目标: 5d forward log return
    target = np.log(gld_close / gld_close.shift(5)).shift(-5)

    # 对齐
    valid = features[feat_cols].notna().all(axis=1) & target.notna()
    feat_df = features.loc[valid, feat_cols]
    target_s = target[valid]
    price_s = gld_close[valid]

    dates = feat_df.index.sort_values()
    n = len(dates)

    print(f"  有效样本: {n}")
    print(f"  日期: {dates[0].date()} ~ {dates[-1].date()}")

    # Walk-forward 参数
    min_train = 1260    # 5年最小训练
    val_size = 252      # 1年验证
    test_size = 126     # 半年测试
    step = 126          # 半年步进
    seq_len = 20        # LSTM 输入序列长度

    # 收集 OOS 预测
    oos_preds = pd.Series(dtype=float)
    oos_actuals = pd.Series(dtype=float)
    fold_results = []

    cutoff = min_train
    fold_id = 0

    while cutoff + test_size <= n:
        fold_id += 1
        train_end = cutoff
        val_start = max(0, train_end - val_size)
        test_end = min(cutoff + test_size, n)

        train_dates = dates[:train_end]
        val_dates = dates[val_start:train_end]
        test_dates = dates[cutoff:test_end]

        X_train = feat_df.loc[train_dates].values
        y_train = target_s.loc[train_dates].values
        X_val = feat_df.loc[val_dates].values
        y_val = target_s.loc[val_dates].values
        X_test = feat_df.loc[test_dates].values
        y_test = target_s.loc[test_dates].values

        print(f"\n  Fold {fold_id}: "
              f"train={train_dates[0].date()}~{train_dates[-1].date()} "
              f"({len(train_dates)}), "
              f"test={test_dates[0].date()}~{test_dates[-1].date()} "
              f"({len(test_dates)})")

        # 训练
        model_kwargs = dict(
            seq_len=seq_len,
            dropout=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=100,
            batch_size=64,
            patience=15,
            model_type=model_type,
        )
        if model_type == "lstm":
            model_kwargs.update(hidden_size=64, num_layers=2)
        else:  # transformer
            model_kwargs.update(
                d_model=64, nhead=4, num_layers=3,
                dim_feedforward=128)
        predictor = DLFairValuePredictor(**model_kwargs)
        predictor.fit(X_train, y_train, X_val, y_val, verbose=False)

        # 预测
        # 需要 seq_len 个历史步来产出第一个预测
        # 所以给测试集前面补上 seq_len-1 个训练集末尾的数据
        n_prefix = seq_len - 1
        prefix_dates = dates[cutoff - n_prefix: cutoff]
        combined_dates = prefix_dates.append(test_dates)
        X_combined = feat_df.loc[combined_dates].values

        preds = predictor.predict(X_combined)
        # preds 长度 = len(combined) - seq_len + 1 = len(test_dates)
        assert len(preds) == len(test_dates), \
            f"preds {len(preds)} != test {len(test_dates)}"

        # IC
        pred_s = pd.Series(preds, index=test_dates)
        actual_s = target_s.loc[test_dates]
        ic, ic_p = stats.spearmanr(preds, actual_s.values)

        # Direction accuracy
        dir_acc = ((preds > 0) == (actual_s.values > 0)).mean()

        # Fair value tracking
        pred_prices = price_s.loc[test_dates].values * np.exp(preds)
        actual_future = gld_close.shift(-5).reindex(test_dates).values
        valid_mask = ~np.isnan(actual_future)
        if valid_mask.sum() > 0:
            mae_pct = np.mean(
                np.abs(pred_prices[valid_mask] - actual_future[valid_mask])
                / actual_future[valid_mask]) * 100
        else:
            mae_pct = np.nan

        print(f"    IC={ic:+.3f} (p={ic_p:.4f}), "
              f"DirAcc={dir_acc:.1%}, MAE%={mae_pct:.1f}%")

        fold_results.append({
            "fold": fold_id,
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "ic": ic,
            "ic_p": ic_p,
            "dir_acc": dir_acc,
            "mae_pct": mae_pct,
            "n_test": len(test_dates),
        })

        oos_preds = pd.concat([oos_preds, pred_s])
        oos_actuals = pd.concat([oos_actuals, actual_s])

        cutoff += step

    # 去重
    oos_preds = oos_preds[~oos_preds.index.duplicated(keep="last")]
    oos_actuals = oos_actuals[~oos_actuals.index.duplicated(keep="last")]

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*60}")
    print("  Walk-Forward 汇总")
    print(f"{'='*60}")

    fold_df = pd.DataFrame(fold_results)
    print(f"\n  Folds: {len(fold_df)}")
    print(f"  OOS 样本: {len(oos_preds)}")
    print(f"  OOS 日期: {oos_preds.index.min().date()} ~ "
          f"{oos_preds.index.max().date()}")

    print(f"\n  {'Fold':>5s} {'期间':>25s} {'IC':>7s} {'DirAcc':>7s} "
          f"{'MAE%':>6s}")
    print(f"  {'-'*55}")
    for _, r in fold_df.iterrows():
        print(f"  {r['fold']:5.0f} "
              f"{r['test_start'].strftime('%Y-%m')}~"
              f"{r['test_end'].strftime('%Y-%m'):>10s} "
              f"{r['ic']:+7.3f} {r['dir_acc']:7.1%} {r['mae_pct']:6.1f}%")

    avg_ic = fold_df["ic"].mean()
    avg_dir = fold_df["dir_acc"].mean()
    avg_mae = fold_df["mae_pct"].mean()
    print(f"  {'均值':>5s} {'':>25s} {avg_ic:+7.3f} {avg_dir:7.1%} "
          f"{avg_mae:6.1f}%")

    # 总体 IC
    overall_ic, overall_p = stats.spearmanr(oos_preds, oos_actuals)
    print(f"\n  总体 OOS IC = {overall_ic:+.3f} (p={overall_p:.6f})")

    # 与 EMA 对比
    print(f"\n{'='*60}")
    print(f"  对比: {model_name} vs EMA(20)")
    print(f"{'='*60}")

    ema20 = gld_close.ewm(span=20).mean()
    ema_pred_ret = np.log(ema20 / gld_close)
    common_idx = oos_preds.index.intersection(ema_pred_ret.dropna().index)

    if len(common_idx) > 100:
        ema_ic, ema_p = stats.spearmanr(
            ema_pred_ret.loc[common_idx],
            oos_actuals.loc[common_idx])
        dl_ic, dl_p = stats.spearmanr(
            oos_preds.loc[common_idx],
            oos_actuals.loc[common_idx])

        print(f"  {model_name} IC = {dl_ic:+.3f} (p={dl_p:.6f})")
        print(f"  EMA       IC = {ema_ic:+.3f} (p={ema_p:.6f})")
        print(f"  提升: {dl_ic - ema_ic:+.3f}")

    # ============================================================
    # 生成公允价格序列
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  生成 {model_name} 公允价格序列")
    print(f"{'='*60}")

    dl_fv = price_s.reindex(oos_preds.index) * np.exp(oos_preds)
    actual_future_price = gld_close.shift(-5).reindex(oos_preds.index)

    valid_fv = dl_fv.notna() & actual_future_price.notna()
    if valid_fv.sum() > 100:
        dev = (dl_fv[valid_fv] - actual_future_price[valid_fv]) \
            / actual_future_price[valid_fv]
        print(f"  {model_name} FV vs 实际(t+5): "
              f"mean偏差={dev.mean():+.2%}, std={dev.std():.2%}")
        print(f"  |偏差| < 3%: {(dev.abs() < 0.03).mean():.1%}")
        print(f"  |偏差| < 5%: {(dev.abs() < 0.05).mean():.1%}")

    # ============================================================
    # 简单回测: 用预测方向做多空
    # ============================================================
    print(f"\n{'='*60}")
    print("  简单回测: 预测方向策略")
    print(f"{'='*60}")

    daily_ret = gld_close.pct_change().reindex(oos_preds.index).fillna(0)

    # 策略: pred > 0 → 持仓 1.0, pred < 0 → 持仓 0.0
    pos_long = (oos_preds > 0).astype(float)
    # 策略: pred > threshold → 1.0, pred < -threshold → 0.0
    threshold = oos_preds.std() * 0.3
    pos_threshold = pd.Series(0.0, index=oos_preds.index)
    pos_threshold[oos_preds > threshold] = 1.0
    pos_threshold[oos_preds < -threshold] = 0.0
    pos_threshold = pos_threshold.ffill()

    strategies = {
        "LSTM Long/Flat": pos_long.shift(1).fillna(0) * daily_ret,
        "LSTM Threshold": pos_threshold.shift(1).fillna(0) * daily_ret,
        "Buy & Hold": daily_ret,
    }

    print(f"\n  {'策略':20s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} "
          f"{'暴露':>6s}")
    print(f"  {'-'*50}")

    for name, sr in strategies.items():
        cum = (1 + sr).cumprod()
        total = cum.iloc[-1] - 1
        n_years = (oos_preds.index[-1] - oos_preds.index[0]).days / 365.25
        cagr = (1 + total) ** (1/max(n_years, 0.01)) - 1
        sharpe = sr.mean() / sr.std() * np.sqrt(252) \
            if sr.std() > 0 else 0
        dd = (cum / cum.cummax() - 1).min()
        if "Long/Flat" in name:
            exp = pos_long.mean()
        elif "Threshold" in name:
            exp = (pos_threshold > 0).mean()
        else:
            exp = 1.0
        print(f"  {name:20s} {cagr:+7.1%} {sharpe:+7.2f} "
              f"{dd:+7.1%} {exp:6.1%}")

    # 保存
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)

    result_df = pd.DataFrame({
        "predicted_5d_return": oos_preds,
        "actual_5d_return": oos_actuals,
        "dl_fair_value": dl_fv,
        "gld_close": price_s.reindex(oos_preds.index),
    })
    out_name = f"dl_{model_type}_fair_value_oos.parquet"
    result_df.to_parquet(os.path.join(out_dir, out_name))
    print(f"\n  结果已保存到 {out_dir}/{out_name}")

    return {
        "model_type": model_type,
        "overall_ic": overall_ic,
        "overall_p": overall_p,
        "avg_ic": avg_ic,
        "avg_dir_acc": avg_dir,
        "avg_mae": avg_mae,
        "fold_df": fold_df,
        "oos_preds": oos_preds,
        "oos_actuals": oos_actuals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both",
                        choices=["lstm", "transformer", "both"])
    args = parser.parse_args()

    if args.model == "both":
        results = {}
        for mt in ["lstm", "transformer"]:
            results[mt] = run_walk_forward(mt)

        # 对比汇总
        print("\n" + "=" * 70)
        print("  LSTM vs Transformer 对比汇总")
        print("=" * 70)
        print(f"\n  {'指标':15s} {'LSTM':>10s} {'Transformer':>12s}")
        print(f"  {'-'*40}")
        for key, label in [("overall_ic", "总体 IC"),
                            ("avg_ic", "平均 Fold IC"),
                            ("avg_dir_acc", "方向准确率"),
                            ("avg_mae", "MAE%")]:
            v1 = results["lstm"][key]
            v2 = results["transformer"][key]
            if key == "avg_dir_acc":
                print(f"  {label:15s} {v1:10.1%} {v2:12.1%}")
            elif key == "avg_mae":
                print(f"  {label:15s} {v1:10.1f}% {v2:11.1f}%")
            else:
                print(f"  {label:15s} {v1:+10.3f} {v2:+12.3f}")

        # 逐 fold 对比
        print(f"\n  {'Fold':>5s} {'LSTM IC':>9s} {'TF IC':>9s} {'Winner':>8s}")
        print(f"  {'-'*35}")
        lstm_wins = 0
        tf_wins = 0
        for i in range(len(results["lstm"]["fold_df"])):
            l_ic = results["lstm"]["fold_df"].iloc[i]["ic"]
            t_ic = results["transformer"]["fold_df"].iloc[i]["ic"]
            winner = "LSTM" if l_ic > t_ic else "TF"
            if l_ic > t_ic:
                lstm_wins += 1
            else:
                tf_wins += 1
            print(f"  {i+1:5d} {l_ic:+9.3f} {t_ic:+9.3f} {winner:>8s}")
        print(f"  {'胜场':>5s} {lstm_wins:>9d} {tf_wins:>9d}")
    else:
        run_walk_forward(args.model)
