"""
V2 周频区间交易回测 (Phase 4A Step 3 redesign)

改进点:
1. 公允价格 = EMA(20) + macro shift, 不再用 CICC Ridge
2. 每周五重算 (真正的周频)
3. 区间宽度由 vol/GVZ/momentum 驱动
4. Regime × band_position 决定仓位

用法:
    conda activate gold
    python src/models/train_weekly_range_v2.py
"""

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
from src.models.regime_classifier import RegimeClassifier
from src.models.weekly_range_signal_v2 import WeeklyRangeSignalV2

warnings.filterwarnings("ignore")


def load_data():
    """加载并对齐所有数据。"""
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    gld_close = gld["Close"].rename("gld_close")

    common = features.index.intersection(gld_close.index)
    features = features.loc[common]
    gld_close = gld_close.loc[common]

    return features, gld_close


def backtest_strategy(signal_df: pd.DataFrame, name: str = "") -> dict:
    """对信号 DataFrame 进行回测, 返回指标字典。"""
    daily_ret = signal_df["gld_close"].pct_change().fillna(0)
    pos = signal_df["position"]
    strat_ret = pos.shift(1).fillna(0) * daily_ret

    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    n_years = (signal_df.index[-1] - signal_df.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) \
        if strat_ret.std() > 0 else 0
    dd = (cum / cum.cummax() - 1).min()
    exposure = (pos > 0).mean()
    trades = (pos.diff().abs() > 0.01).sum()
    avg_hold = len(signal_df) * exposure / max(trades, 1)

    return {
        "name": name,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": dd,
        "total_return": total,
        "exposure": exposure,
        "trades": trades,
        "avg_hold_days": avg_hold,
        "cum_returns": cum,
        "strat_returns": strat_ret,
        "positions": pos,
    }


def run_evaluation():
    features, gld_close = load_data()

    print("=" * 70)
    print("  Phase 4A Step 3 V2: 自适应周频区间交易")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Regime
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime_clf = RegimeClassifier()
    regime = regime_clf.classify_simple(features[feat_cols])

    # 提取因子
    tw_usd = features["tw_usd"] if "tw_usd" in features.columns else None
    ry = features["real_yield_10y"] \
        if "real_yield_10y" in features.columns else None
    gvz = features["gvz"] if "gvz" in features.columns else None
    ret_20d = features["ret_20d"] if "ret_20d" in features.columns else None

    oos_start = "2016-01-01"
    oos_idx = gld_close.index[gld_close.index >= oos_start]
    regime_oos = regime.reindex(oos_idx)

    print(f"  全样本: {features.index.min().date()} ~ "
          f"{features.index.max().date()} ({len(features)})")
    print(f"  OOS: {oos_idx[0].date()} ~ {oos_idx[-1].date()} "
          f"({len(oos_idx)})")

    # ============================================================
    # 1. Regime 分布
    # ============================================================
    print(f"\n{'='*60}")
    print("  1. Regime 分布 (OOS)")
    print(f"{'='*60}")
    for r in ["Bull", "Mixed", "Bear"]:
        n = (regime_oos == r).sum()
        print(f"  {r:6s}: {n:5d} 天 ({n/len(regime_oos):.1%})")

    # ============================================================
    # 2. 参数配置对比
    # ============================================================
    print(f"\n{'='*60}")
    print("  2. V2 策略参数扫描")
    print(f"{'='*60}")

    configs = [
        {
            "name": "V2-A: 标准配置",
            "params": dict(buy_zone=0.20, sell_zone=0.85,
                           bull_min=0.5, bull_default=0.7,
                           mixed_buy=0.5, mixed_default=0.2),
        },
        {
            "name": "V2-B: Bull不减仓",
            "params": dict(buy_zone=0.20, sell_zone=0.85,
                           bull_min=1.0, bull_default=1.0,
                           mixed_buy=0.5, mixed_default=0.2),
        },
        {
            "name": "V2-C: 保守(窄触发)",
            "params": dict(buy_zone=0.15, sell_zone=0.90,
                           bull_min=0.5, bull_default=0.7,
                           mixed_buy=0.3, mixed_default=0.1),
        },
        {
            "name": "V2-D: 积极(宽触发)",
            "params": dict(buy_zone=0.30, sell_zone=0.75,
                           bull_min=0.3, bull_default=0.5,
                           mixed_buy=0.5, mixed_default=0.3),
        },
    ]

    best_result = None
    all_results = []

    print(f"\n  {'策略':25s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} "
          f"{'暴露':>6s} {'交易':>5s} {'持仓':>5s}")
    print(f"  {'-'*65}")

    for cfg in configs:
        signal_gen = WeeklyRangeSignalV2(**cfg["params"])
        signal = signal_gen.generate(
            regime_oos, gld_close,
            tw_usd=tw_usd, real_yield_10y=ry,
            gvz=gvz, ret_20d=ret_20d,
        )
        result = backtest_strategy(signal, cfg["name"])
        all_results.append(result)

        print(f"  {cfg['name']:25s} {result['cagr']:+7.1%} "
              f"{result['sharpe']:+7.2f} {result['max_dd']:+7.1%} "
              f"{result['exposure']:6.1%} {result['trades']:5d} "
              f"{result['avg_hold_days']:5.0f}d")

        if best_result is None or result["sharpe"] > best_result["sharpe"]:
            best_result = result

    # 基准
    daily_ret = gld_close.reindex(oos_idx).pct_change().fillna(0)
    benchmarks = {
        "Pure Regime (Bull=1)":
            (regime_oos == "Bull").astype(float).shift(1).fillna(0)
            * daily_ret,
        "Buy & Hold": daily_ret,
    }

    print()
    for bname, bret in benchmarks.items():
        cum = (1 + bret).cumprod()
        total = cum.iloc[-1] - 1
        n_years = (oos_idx[-1] - oos_idx[0]).days / 365.25
        cagr = (1 + total) ** (1/max(n_years, 0.01)) - 1
        sharpe = bret.mean() / bret.std() * np.sqrt(252) \
            if bret.std() > 0 else 0
        dd = (cum / cum.cummax() - 1).min()
        exp = (regime_oos == "Bull").mean() if "Regime" in bname else 1.0
        print(f"  {bname:25s} {cagr:+7.1%} {sharpe:+7.2f} "
              f"{dd:+7.1%} {exp:6.1%}")

    # ============================================================
    # 3. 最佳策略年度明细
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  3. 最佳策略年度明细: {best_result['name']}")
    print(f"{'='*60}")

    strat_ret = best_result["strat_returns"]
    regime_ret = benchmarks["Pure Regime (Bull=1)"]
    bh_ret = benchmarks["Buy & Hold"]

    print(f"\n  {'年份':>6s} {'V2':>9s} {'Regime':>9s} {'B&H':>9s} "
          f"{'暴露':>7s} {'交易':>5s} {'Bull%':>6s}")
    print(f"  {'-'*55}")

    pos = best_result["positions"]
    for year in sorted(oos_idx.year.unique()):
        yr = oos_idx.year == year
        v2_yr = (1 + strat_ret[yr]).cumprod().iloc[-1] - 1
        reg_yr = (1 + regime_ret[yr]).cumprod().iloc[-1] - 1
        bh_yr = (1 + bh_ret[yr]).cumprod().iloc[-1] - 1
        exp = (pos[yr] > 0).mean()
        trades = (pos[yr].diff().abs() > 0.01).sum()
        bull_pct = (regime_oos[yr] == "Bull").mean()
        print(f"  {year:>6d} {v2_yr:+9.1%} {reg_yr:+9.1%} "
              f"{bh_yr:+9.1%} {exp:7.1%} {trades:5d} {bull_pct:6.1%}")

    # ============================================================
    # 4. Band Position 区分力验证
    # ============================================================
    print(f"\n{'='*60}")
    print("  4. Regime x Band Position 未来5d收益")
    print(f"{'='*60}")

    # 用最佳策略的信号重新生成
    best_cfg = [c for c in configs
                if c["name"] == best_result["name"]][0]
    sig_gen = WeeklyRangeSignalV2(**best_cfg["params"])
    signal = sig_gen.generate(
        regime_oos, gld_close,
        tw_usd=tw_usd, real_yield_10y=ry,
        gvz=gvz, ret_20d=ret_20d,
    )

    fwd_5d = gld_close.pct_change(5).shift(-5).reindex(oos_idx)
    signal["fwd_ret_5d"] = fwd_5d

    print(f"\n  {'Regime':>7s} {'Band Zone':>15s} {'5d均收益':>9s} "
          f"{'胜率':>6s} {'Sharpe':>7s} {'样本':>6s}")
    print(f"  {'-'*55}")

    for reg_val in ["Bull", "Mixed", "Bear"]:
        for label, lo, hi in [("下界 (bp<0.20)", -999, 0.20),
                               ("偏低 (0.20-0.45)", 0.20, 0.45),
                               ("中间 (0.45-0.70)", 0.45, 0.70),
                               ("偏高 (0.70-0.85)", 0.70, 0.85),
                               ("上界 (bp>0.85)", 0.85, 999)]:
            mask = ((signal["regime"] == reg_val)
                    & (signal["band_position"] >= lo)
                    & (signal["band_position"] < hi))
            rets = signal.loc[mask, "fwd_ret_5d"].dropna()
            if len(rets) < 15:
                continue
            ann_s = rets.mean() / rets.std() * np.sqrt(252/5) \
                if rets.std() > 0 else 0
            print(f"  {reg_val:>7s} {label:>15s} {rets.mean():+9.3%} "
                  f"{(rets>0).mean():6.1%} {ann_s:+7.2f} {len(rets):6d}")
        print()

    # t-test: Bull 下界 vs 上界
    bull_data = signal[signal["regime"] == "Bull"]
    low_rets = bull_data.loc[
        bull_data["band_position"] < 0.30, "fwd_ret_5d"].dropna()
    high_rets = bull_data.loc[
        bull_data["band_position"] > 0.70, "fwd_ret_5d"].dropna()
    if len(low_rets) > 15 and len(high_rets) > 15:
        t, p = stats.ttest_ind(low_rets, high_rets)
        print(f"  Bull: 下界 vs 上界 t-test: t={t:.2f}, p={p:.4f} "
              f"{'***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''}")
        print(f"    下界 (bp<0.30): mean={low_rets.mean():+.3%}, n={len(low_rets)}")
        print(f"    上界 (bp>0.70): mean={high_rets.mean():+.3%}, n={len(high_rets)}")

    # ============================================================
    # 5. 公允价格追踪能力
    # ============================================================
    print(f"\n{'='*60}")
    print("  5. 公允价格追踪能力")
    print(f"{'='*60}")

    fv = signal["fair_value"]
    actual = signal["gld_close"]
    deviation = (actual - fv) / fv

    print(f"  偏离度: mean={deviation.mean():+.2%}, std={deviation.std():.2%}")
    print(f"  范围: [{deviation.min():.2%}, {deviation.max():.2%}]")
    print(f"  中位数: {deviation.median():+.2%}")
    print(f"  |偏离| < 3%: {(deviation.abs() < 0.03).mean():.1%}")
    print(f"  |偏离| < 5%: {(deviation.abs() < 0.05).mean():.1%}")

    # 公允价格在真实价格上方/下方的比例
    above = (fv > actual).mean()
    below = (fv < actual).mean()
    print(f"  公允 > 实际: {above:.1%}")
    print(f"  公允 < 实际: {below:.1%}")

    # 近一年
    recent = signal.index >= (signal.index[-1] - pd.Timedelta(days=365))
    if recent.sum() > 50:
        dev_recent = deviation[recent]
        print(f"\n  近一年偏离度: mean={dev_recent.mean():+.2%}, "
              f"std={dev_recent.std():.2%}")
        print(f"  近一年 |偏离| < 3%: "
              f"{(dev_recent.abs() < 0.03).mean():.1%}")

    # ============================================================
    # 6. 区间宽度统计
    # ============================================================
    print(f"\n{'='*60}")
    print("  6. 区间宽度统计")
    print(f"{'='*60}")

    hw = signal["half_width"]
    band_width_pct = hw * 2 * 100

    print(f"  区间宽度 (2×half_width): "
          f"mean={band_width_pct.mean():.1f}%, "
          f"median={band_width_pct.median():.1f}%")
    print(f"  范围: [{band_width_pct.min():.1f}%, {band_width_pct.max():.1f}%]")

    # 按年
    for year in sorted(oos_idx.year.unique()):
        yr = oos_idx.year == year
        yr_hw = band_width_pct[yr]
        print(f"  {year}: mean={yr_hw.mean():.1f}%, "
              f"min={yr_hw.min():.1f}%, max={yr_hw.max():.1f}%")

    # ============================================================
    # 保存结果
    # ============================================================
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    signal.to_parquet(os.path.join(out_dir, "weekly_range_v2_signal.parquet"))
    print(f"\n  结果已保存到 {out_dir}/weekly_range_v2_signal.parquet")

    return signal, best_result


if __name__ == "__main__":
    run_evaluation()
