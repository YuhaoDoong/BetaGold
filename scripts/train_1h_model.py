"""训练 1h 多粒度区间预测模型.

用法:
    conda activate gold
    python scripts/train_1h_model.py

流程:
    1. 加载 1h 数据 (GLD + 跨市场)
    2. 加载日线 Regime (复用 v1.0 模型)
    3. 构建多粒度特征 + 目标
    4. Walk-forward 训练 (3种子集成 + Conformal校准)
    5. OOS 预测评估
    6. 保存结果
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.features_1h import build_dataset
from core.dl_range_1h import DLRangePredictor1h


def load_1h_csv(path):
    """加载 1h CSV."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_daily_regime():
    """从 v1.0 日线模型加载 Regime 和 RV percentile."""
    from core.data import load_config, load_features, load_gld
    from core.regime import RegimeClassifier
    from core.signals import compute_rv_pctile

    cfg = load_config()
    features = load_features(cfg)
    gld = load_gld(cfg)

    common = features.index.intersection(gld.index)
    features = features.loc[common]

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    rv_10d = features["rv_10d"] if "rv_10d" in features.columns else None
    rv_pctile = compute_rv_pctile(rv_10d)

    return regime, rv_pctile


def main():
    t0 = time.time()
    data_dir = os.path.join(ROOT, "..", "Gold", "data", "raw", "market")
    data_dir = os.path.normpath(data_dir)

    print("=" * 60)
    print("1h 多粒度区间预测模型训练")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/5] 加载 1h 数据...")
    gld_1h = load_1h_csv(os.path.join(data_dir, "gld_1h.csv"))
    gc_1h = load_1h_csv(os.path.join(data_dir, "gc_1h.csv"))
    dxy_1h = load_1h_csv(os.path.join(data_dir, "dxy_1h.csv"))
    vix_1h = load_1h_csv(os.path.join(data_dir, "vix_1h.csv"))
    slv_1h = load_1h_csv(os.path.join(data_dir, "slv_1h.csv"))
    tlt_1h = load_1h_csv(os.path.join(data_dir, "tlt_1h.csv"))

    print(f"  GLD 1h: {len(gld_1h)} bars ({gld_1h.index[0]} ~ {gld_1h.index[-1]})")
    for name, df in [("GC", gc_1h), ("DXY", dxy_1h), ("VIX", vix_1h),
                      ("SLV", slv_1h), ("TLT", tlt_1h)]:
        if df is not None:
            print(f"  {name} 1h: {len(df)} bars")

    # ── 日线 Regime ──
    print("\n[2/5] 加载日线 Regime...")
    regime, rv_pctile = load_daily_regime()
    print(f"  Regime: {len(regime)} days, latest={regime.iloc[-1]}")

    # ── 构建特征 ──
    print("\n[3/5] 构建多粒度特征...")
    horizons = (7, 35)  # 7h≈1天, 35h≈5天
    features, targets = build_dataset(
        gld_1h, gc_1h, dxy_1h, vix_1h, slv_1h, tlt_1h,
        regime_series=regime, daily_rv_pctile=rv_pctile,
        horizons=horizons,
    )

    # 清理
    # Drop rows where all features are NaN (warmup period)
    valid_mask = features.notna().sum(axis=1) > features.shape[1] * 0.5
    features = features[valid_mask]
    targets = targets.reindex(features.index)

    print(f"  特征: {features.shape[1]} 维 × {len(features)} 行")
    print(f"  特征列: {list(features.columns[:10])} ... ({features.shape[1]} total)")

    # NaN 填充
    features = features.ffill().bfill().fillna(0)

    # RV scale
    rv_scale = features["rv_10h"].values.copy()
    rv_scale = np.clip(rv_scale, 0.5, None)

    # ── 训练 ──
    print("\n[4/5] 训练模型...")

    # Train/test split: 后 25% 作为 OOS 测试
    n = len(features)
    train_end = int(n * 0.75)
    val_size = 350   # ~50天
    cal_size = 175   # ~25天

    print(f"  总样本: {n}")
    print(f"  训练截止: {features.index[train_end]} (idx={train_end})")
    print(f"  测试: {features.index[train_end]} ~ {features.index[-1]}")
    print(f"  Val: {val_size} bars, Cal: {cal_size} bars")

    targets_list = []
    for nh in horizons:
        u = targets[f"fwd_{nh}h_upper_pct"].values
        l = targets[f"fwd_{nh}h_lower_pct"].values
        targets_list.append((u, l))

    predictor = DLRangePredictor1h(
        seq_len=48,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        epochs=150,
        batch_size=64,
        patience=20,
        q_upper=0.85,
        q_lower=0.15,
        n_ensemble=3,
        cal_target_cov=0.80,
        horizons=horizons,
    )

    results = predictor.fit_predict(
        features.values, targets_list, rv_scale,
        features.index, train_end, val_size, cal_size,
    )

    # ── 评估 ──
    print("\n[5/5] OOS 评估...")
    for h_idx, nh in enumerate(horizons):
        pred_df = results[h_idx]
        actual_u = targets[f"fwd_{nh}h_upper_pct"].reindex(pred_df.index)
        actual_l = targets[f"fwd_{nh}h_lower_pct"].reindex(pred_df.index)
        pred_u = pred_df[f"pred_{nh}h_upper_pct"]
        pred_l = pred_df[f"pred_{nh}h_lower_pct"]

        valid = actual_u.notna() & actual_l.notna()
        au, al = actual_u[valid], actual_l[valid]
        pu, pl = pred_u[valid], pred_l[valid]

        # Coverage: actual within [pred_lower, pred_upper]
        covered = ((au <= pu) & (al >= pl)).mean()

        # IC (rank correlation)
        ic_u = au.corr(pu, method="spearman")
        ic_l = al.corr(pl, method="spearman")

        # Width
        width = (pu - pl).mean()

        print(f"\n  {nh}h horizon ({len(pu)} OOS samples):")
        print(f"    Coverage: {covered:.1%} (target: 80%)")
        print(f"    IC upper: {ic_u:.3f}, IC lower: {ic_l:.3f}")
        print(f"    Mean width: {width:.2f}%")
        print(f"    Pred range: [{pl.mean():.2f}%, {pu.mean():.2f}%]")

    # ── 保存 ──
    save_dir = os.path.join(ROOT, "..", "Gold", "data", "models")
    os.makedirs(save_dir, exist_ok=True)

    for h_idx, nh in enumerate(horizons):
        path = os.path.join(save_dir, f"dl_range_1h_{nh}h_oos.parquet")
        results[h_idx].to_parquet(path)
        print(f"\n  Saved: {path} ({len(results[h_idx])} rows)")

    # 保存合并版
    merged = pd.concat(results.values(), axis=1)
    merged_path = os.path.join(save_dir, "dl_range_1h_oos.parquet")
    merged.to_parquet(merged_path)
    print(f"  Saved merged: {merged_path}")

    elapsed = time.time() - t0
    print(f"\n完成! 耗时 {elapsed:.0f}s")


if __name__ == "__main__":
    main()
